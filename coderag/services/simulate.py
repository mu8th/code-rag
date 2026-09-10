"""Offline simulation engine: real retrieval, rule-based synthesis, no models.

The demo is meant to run 24/7 on a machine that does not keep a local LLM
loaded, so this module provides a fully offline path through the pipeline:

- :func:`local_embed` replaces the Ollama embedding endpoint with a
  deterministic feature-hashing vectorizer (the "hashing trick"). Tokens are
  hashed into a fixed number of signed buckets and the result is L2-normalized.
  It is a genuine similarity model for code: two passages that share identifiers
  and keywords get a higher cosine score than ones that do not, so retrieval
  still ranks the right symbols first. No network, no model weights, and the
  same text always yields the same vector on any machine.

- :func:`synthesize_local` replaces the LLM step with a deterministic template
  that composes a grounded answer from the retrieved passages alone: it names
  the top-ranked symbol and quotes a short excerpt from each cited passage.
  The answer is exactly as good as the retrieval, never better, which keeps the
  demo honest about what simulation mode can and cannot do.

Everything here is standard library plus numpy; nothing in this module touches
the network.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from ..models import Chunk

#: Dimensionality of the simulated embedding space.
EMBED_DIM = 512

#: Token pattern: runs of lowercase letters, digits, and underscores, so Python
#: identifiers like ``cosine_similarity`` stay intact as single tokens.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/identifier tokens, their underscore parts, and bigrams.

    Splitting ``retrieve_items`` into ``retrieve`` and ``items`` as well lets a
    natural-language query share tokens with snake_case identifiers, which is
    where most of the retrieval signal for code comes from.
    """
    raw = _TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    for tok in raw:
        tokens.append(tok)
        if "_" in tok:
            tokens.extend(p for p in tok.split("_") if p)
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
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in _tokenize(text):
            index, sign = _bucket(token)
            matrix[i, index] += sign
        norm = np.linalg.norm(matrix[i])
        if norm > 0:
            matrix[i] /= norm
    return matrix


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


def synthesize_local(question: str, chunks: Sequence[Chunk]) -> str:
    """Compose a grounded, cited answer from the retrieved passages alone.

    Args:
        question: The user's question (kept for interface symmetry; the answer
            is derived from the passages, not from free generation).
        chunks: Retrieved passages, most relevant first.

    Returns:
        A short deterministic answer naming the top-ranked symbol and quoting
        one line from each cited passage.
    """
    if not chunks:
        return "No relevant passages were found in the indexed codebase."
    top = chunks[0]
    parts = [
        (f"Top match for this question is {top.path}::{top.symbol} "
         f"(lines {top.start_line}-{top.end_line}). Grounded summary from the "
         f"retrieved context:")
    ]
    for i, chunk in enumerate(chunks[:3], start=1):
        line = _citation_line(chunk.text)
        if i == 1:
            parts.append(f"[1] {line or chunk.symbol}")
        elif line:
            parts.append(f"[{i}] {chunk.path}::{chunk.symbol}: {line}")
        else:
            parts.append(f"[{i}] {chunk.path}::{chunk.symbol}")
    return "\n".join(parts)
