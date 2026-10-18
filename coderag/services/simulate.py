"""Offline simulation engine: real retrieval, extractive synthesis, no models.

The demo is meant to run 24/7 on a machine that does not keep a local LLM
loaded, so this module provides a fully offline path through the pipeline:

- :class:`LexicalIndex` ranks passages with BM25 over three weighted fields:
  the file path and symbol name, the docstring and comment prose, and every
  code term. Scores are normalized to the share of the question matched, from
  0 to 1, which gives the pipeline's relevance floor a stable meaning across
  questions.

- :func:`synthesize_local` replaces the LLM step with extractive synthesis: it
  quotes the docstring and comment sentences from the retrieved passages that
  share the most rare terms with the question, and cites each one. Every
  sentence in the answer is copied from the code it cites, so the answer is
  exactly as good as the retrieval, never better.

- :func:`local_embed` is a deterministic feature-hashing vectorizer (the
  "hashing trick") for callers that need offline vectors in the embedding
  contract: the same text always yields the same L2-normalized vector, with no
  network and no model weights.

Everything here is standard library plus numpy; nothing in this module touches
the network.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise

import numpy as np

from ..models import Chunk

#: Dimensionality of the simulated embedding space.
EMBED_DIM = 512

#: Token pattern: runs of lowercase letters, digits, and underscores, so Python
#: identifiers like ``cosine_similarity`` stay intact as single tokens.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


#: English function words and Python boilerplate. They appear in nearly every
#: question and every chunk, so they carry no signal about *which* code is
#: relevant and only dilute the cosine score.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been but by can could did do does doing for from had
    has have how i if in into is it its may me my not of on or our should so
    than that the their them then there these they this those to us was we were
    what when where which while who why will with would you your
    def class self cls return none true false import pass elif else
    str int float bool list dict tuple set any
    """.split()
)

_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")


_SUFFIXES = ("ations", "ation", "ions", "ion", "ing", "ies", "es", "ed", "s")


def _stem(token: str) -> str:
    """Strip a common English suffix so ``chunks``/``chunking``/``chunk`` meet.

    Deliberately crude (no dictionary): it only needs to be consistent, because
    queries and passages go through the same function. ``citations``,
    ``citing`` and ``cites`` all become ``cit``; ``embedding`` and ``embeds``
    become ``embed``.
    """
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            if suffix == "ies":
                return stem + "y"
            if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                stem = stem[:-1]  # embedd -> embed
            return stem
    return token


def _terms(text: str) -> list[str]:
    """Content terms of ``text``: lowercased, split, stopword-free, stemmed.

    ``VectorIndex`` and ``retrieve_items`` are split into their parts as well
    as kept whole, so a natural-language query can meet a code identifier.
    """
    spaced = _CAMEL_RE.sub(r"\1_\2", text)
    terms: list[str] = []
    for tok in _TOKEN_RE.findall(spaced.lower()):
        parts = [p for p in tok.split("_") if p] if "_" in tok else []
        for word in [tok, *parts]:
            if word in STOPWORDS or len(word) < 2 or word.isdigit():
                continue
            terms.append(_stem(word))
    return terms


#: Bigrams add phrase signal ("vector index") but are noisier than single
#: terms, so they count at a fraction of a term's weight.
_BIGRAM_WEIGHT = 0.5


def _tokenize(text: str) -> list[str]:
    """Content terms plus adjacent-term bigrams (phrase signal)."""
    tokens = _terms(text)
    bigrams = [f"{a}_{b}" for a, b in pairwise(tokens)]
    return tokens + bigrams


def _bucket(token: str) -> tuple[int, float]:
    """Map a token to (bucket index, sign) using a stable hash.

    Python's built-in ``hash`` is salted per process for strings, so it would
    make embeddings non-reproducible across runs; hashlib is stable.
    """
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % EMBED_DIM
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


def local_embed(texts: Sequence[str]) -> np.ndarray:
    """Deterministically embed ``texts`` offline (no endpoint, no model).

    Args:
        texts: Strings to embed (may be empty).

    Returns:
        A numpy matrix of shape ``(len(texts), EMBED_DIM)`` with L2-normalized
        rows. An empty input returns an empty ``(0, 0)`` array so callers can
        branch on shape, matching the Ollama embedder's contract.
    """
    return _hash_vectors(texts, None)


def _hash_vectors(
    texts: Sequence[str], idf: Mapping[str, float] | None
) -> np.ndarray:
    """Feature-hash ``texts`` with sublinear term frequency and optional weights."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for token, count in Counter(_tokenize(text)).items():
            weight = 1.0 + math.log(count)
            if "_" in token:
                weight *= _BIGRAM_WEIGHT
            if idf is not None:
                weight *= idf.get(token, idf.get("", 1.0))
            index, sign = _bucket(token)
            matrix[i, index] += sign * weight
        norm = np.linalg.norm(matrix[i])
        if norm > 0:
            matrix[i] /= norm
    return matrix


class LexicalIndex:
    """BM25 index over a chunk's identifiers, docstrings and comments.

    Natural-language questions meet code mostly through names and prose, so
    each passage is indexed as three weighted fields: its location (file path
    and symbol name, x3), its docstring and comment text (x2), and every term
    in the code (x1). Scores are BM25 divided by the best score the question
    could reach, so they read as "share of the question matched" (0 to 1) and a
    fixed relevance floor means the same thing for every question.

    Args:
        chunks: The passages to index.
        k1: BM25 term-frequency saturation.
        b: BM25 length normalization.
    """

    def __init__(
        self, chunks: Sequence[Chunk], k1: float = 1.2, b: float = 0.75
    ) -> None:
        self._k1 = k1
        self._b = b
        self._docs: list[Counter[str]] = []
        for chunk in chunks:
            location = f"{chunk.path.rsplit('.', 1)[0]} {chunk.symbol}"
            prose = " ".join(_explanation_text(chunk.text))
            terms = _terms(location) * 3 + _terms(prose) * 2 + _terms(chunk.text)
            self._docs.append(Counter(terms))
        lengths = [sum(d.values()) for d in self._docs]
        self._lengths = np.array(lengths, dtype=np.float64)
        self._avg_len = float(self._lengths.mean()) if len(self._docs) else 0.0
        doc_freq: Counter[str] = Counter()
        for doc in self._docs:
            doc_freq.update(doc.keys())
        n = len(self._docs)
        #: Inverse document frequency per term (BM25+ style, never negative).
        self.idf: dict[str, float] = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    @property
    def size(self) -> int:
        """Number of indexed passages."""
        return len(self._docs)

    def search(self, question: str, k: int) -> list[tuple[int, float]]:
        """Return the top-``k`` ``(position, score)`` pairs for ``question``.

        Scores are in ``[0, 1]``: 1 means every question term matched about as
        strongly as BM25 allows; 0 means no question term occurs in the code.
        """
        if not self._docs or k <= 0:
            return []
        query = list(dict.fromkeys(_terms(question)))
        terms = [t for t in query if t in self.idf]
        # A term the code never uses still counts toward the ceiling (at the
        # rarest-term weight), so an off-topic question that happens to share
        # one common word with the code cannot score as a full match.
        unseen = math.log(1 + (len(self._docs) + 0.5) / 0.5)
        ceiling = sum(self.idf.get(t, unseen) for t in query) * (self._k1 + 1)
        scores = np.zeros(len(self._docs), dtype=np.float64)
        if ceiling:
            norm = self._k1 * (1 - self._b + self._b * self._lengths / self._avg_len)
            for term in terms:
                counts = [doc.get(term, 0) for doc in self._docs]
                tf = np.array(counts, dtype=np.float64)
                scores += self.idf[term] * tf * (self._k1 + 1) / (tf + norm)
            scores /= ceiling
        k = min(k, len(scores))
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(pos), float(scores[pos])) for pos in top]


_LABEL_RE = re.compile(r"^[A-Za-z_]+:$")


def _citation_line(text: str) -> str:
    """Pick the line that best represents a chunk for display.

    Prefers the symbol's signature (``def``/``class`` line); falls back to the
    first line that is not blank, a comment, a docstring delimiter, or a bare
    docstring label such as ``Args:``.
    """
    fallback = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("def ", "class ")):
            if len(line) > 100:
                return line[:100].rstrip() + "…"
            return line
        if line.startswith(("@", '"""', "'''")) or _LABEL_RE.match(line):
            continue
        if not fallback:
            fallback = line[:100].rstrip() + ("…" if len(line) > 100 else "")
    return fallback


_DOCSTRING_RE = re.compile(r'("""|\'\'\')(.*?)\1', re.DOTALL)
_SECTION_RE = re.compile(
    r"^(Args|Arguments|Returns|Yields|Raises|Attributes|Example|Examples|Note|Notes):$"
)
_BULLET_RE = re.compile(r"^(?:[-*+]|\d+\.)\s+")
_PARAM_RE = re.compile(r"^\*{0,2}\w+(\s*\([^)]*\))?:\s+")
#: Sphinx cross-reference roles (:class:`Source`) read as plain code spans.
_ROLE_RE = re.compile(r":(?:class|func|meth|mod|attr|data|exc):(`[^`]+`)")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`(])")

#: Bounds that keep a simulated answer readable.
_MAX_SENTENCES = 3
_MAX_PER_CHUNK = 2
_MAX_SENTENCE_CHARS = 220
#: A sentence must score at least this share of the best sentence to be quoted,
#: so a weak one-word overlap never pads a strong answer.
_RELATIVE_FLOOR = 0.6


def _explanation_text(text: str) -> list[str]:
    """Docstring bodies and comment lines of a chunk, unsplit."""
    parts = [body for _, body in _DOCSTRING_RE.findall(text)]
    parts += [
        line.strip().lstrip("#:").strip()
        for line in text.splitlines()
        if line.strip().startswith("#") and not line.strip().startswith("#!")
    ]
    return parts


def _explanations(text: str) -> list[str]:
    """Return the prose sentences a chunk contains: docstrings, then comments.

    Google-style section headers (``Args:``) are dropped and parameter labels
    (``k: How many...``) are stripped, so what remains reads as sentences.
    """
    paragraphs: list[str] = []
    for _, body in _DOCSTRING_RE.findall(text):
        current: list[str] = []
        for raw in body.splitlines():
            line = raw.strip()
            if not line or _SECTION_RE.match(line):
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                continue
            if (_PARAM_RE.match(line) or _BULLET_RE.match(line)) and current:
                paragraphs.append(" ".join(current))
                current = []
            current.append(_PARAM_RE.sub("", _BULLET_RE.sub("", line)))
        if current:
            paragraphs.append(" ".join(current))
    comment: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") and not line.startswith("#!"):
            body = line.lstrip("#:").strip()
            if body:
                comment.append(body)
                continue
        if comment:
            paragraphs.append(" ".join(comment))
            comment = []
    if comment:
        paragraphs.append(" ".join(comment))

    sentences: list[str] = []
    for para in paragraphs:
        para = _ROLE_RE.sub(r"\1", para.replace("``", "`"))
        for sentence in _SENTENCE_END_RE.split(para):
            sentence = sentence.strip()
            if len(sentence) < 20 or len(_terms(sentence)) < 3:
                continue
            if len(sentence) > _MAX_SENTENCE_CHARS:
                sentence = sentence[:_MAX_SENTENCE_CHARS].rsplit(" ", 1)[0] + "…"
            elif sentence[-1] not in ".!?…":
                sentence += "."
            sentences.append(sentence)
    return sentences


def _location(chunk: Chunk) -> str:
    if chunk.symbol in {"<module>", "<unparsed>"}:
        return f"the `{chunk.path}` module"
    lines = f"lines {chunk.start_line}-{chunk.end_line}"
    return f"`{chunk.symbol}` in {chunk.path} ({lines})"


def synthesize_local(
    question: str,
    chunks: Sequence[Chunk],
    idf: Mapping[str, float] | None = None,
) -> str:
    """Compose a grounded, cited answer by extracting sentences from passages.

    Args:
        question: The user's question; its content terms select sentences.
        chunks: Retrieved passages, most relevant first.
        idf: Optional term weights (from :class:`LexicalIndex`); rare terms
            then count for more when matching sentences to the question.

    Returns:
        A short answer. It names the top passage, then quotes up to three
        docstring or comment sentences that share terms with the question, each
        followed by the ``[n]`` of the passage it came from. When no sentence
        matches, it quotes the top passage's signature line instead.
    """
    if not chunks:
        return "No relevant passages were found in the indexed codebase."
    weights = idf or {}
    default = weights.get("", 1.0)
    query = set(_terms(question))
    # With three or more content terms, one shared word is usually incidental.
    needed = 2 if len(query) >= 3 else 1

    scored: list[tuple[float, int, int, str]] = []
    for rank, chunk in enumerate(chunks[:4]):
        for order, sentence in enumerate(_explanations(chunk.text)):
            shared = query & set(_terms(sentence))
            if len(shared) < needed:
                continue
            score = sum(weights.get(t, default) for t in shared)
            score *= 1.0 / (1.0 + 0.35 * rank)
            scored.append((score, rank, order, sentence))

    chosen: list[tuple[int, int, str]] = []
    per_chunk: Counter[int] = Counter()
    best = max((x[0] for x in scored), default=0.0)
    for score, rank, order, sentence in sorted(scored, key=lambda x: -x[0]):
        if score < _RELATIVE_FLOOR * best:
            break
        if len(chosen) >= _MAX_SENTENCES or per_chunk[rank] >= _MAX_PER_CHUNK:
            continue
        if any(sentence == c[2] for c in chosen):
            continue
        chosen.append((rank, order, sentence))
        per_chunk[rank] += 1

    top = chunks[0]
    answer = f"The closest match is {_location(top)} [1]."
    if not chosen:
        signature = _citation_line(top.text)
        if signature:
            answer += f" Its definition reads `{signature}`."
    for rank, _, sentence in sorted(chosen):
        answer += f" {sentence} [{rank + 1}]"
    cited = {rank for rank, _, _ in chosen}
    extra = [
        f"[{i + 1}] {_location(c)}"
        for i, c in enumerate(chunks[1:4], start=1)
        if i not in cited
    ]
    if extra:
        answer += " See also " + "; ".join(extra) + "."
    return answer
