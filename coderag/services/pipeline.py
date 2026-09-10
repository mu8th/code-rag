"""The end-to-end RAG pipeline: ingest, embed, retrieve, synthesize.

:class:`CodeRAG` is the single object callers interact with. It builds and
caches an in-memory vector index on first use, so repeated questions against
the same codebase pay the embedding cost only once. The embed and synthesize
steps are injected (with sensible local defaults) so the whole pipeline can be
exercised in tests without a live model.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence

import numpy as np

from ..config import Settings, get_settings
from ..models import Chunk, RagResult, Source
from . import embed as embed_mod
from . import ingestion
from . import simulate as simulate_mod
from . import synthesize as synthesize_mod
from .retrieval import VectorIndex

#: An embedder maps a sequence of strings to an (N, dim) float32 matrix.
Embedder = Callable[[Sequence[str]], np.ndarray]
#: A synthesizer maps (question, chunks) to an answer string.
Synthesizer = Callable[[str, Sequence[Chunk]], str]

#: Number of leading lines kept for a source citation snippet.
_SNIPPET_LINES = 3
#: Maximum characters kept in a citation snippet.
_SNIPPET_CHARS = 240


class CodeRAG:
    """A local RAG assistant over a Python codebase.

    Args:
        settings: Runtime configuration. Defaults to :func:`get_settings`.
        embedder: Optional override for the embedder (used in tests).
        synthesizer: Optional override for the answer synthesizer (used in tests).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder = embedder or self._default_embedder
        self._synthesizer = synthesizer or self._default_synthesizer
        self._chunks: list[Chunk] = []
        self._index: VectorIndex | None = None
        self._index_lock = threading.Lock()
        #: Model id that answered the last query (set by the default
        #: synthesizer so callers can report which endpoint was used).
        self.last_model: str | None = None

    # -- index lifecycle -------------------------------------------------

    def _default_embedder(self, texts: Sequence[str]) -> np.ndarray:
        if self.settings.simulate:
            return simulate_mod.local_embed(texts)
        return embed_mod.embed_texts(
            texts, self.settings.embed_endpoint, self.settings.embed_model
        )

    def _default_synthesizer(self, question: str, chunks: Sequence[Chunk]) -> str:
        if self.settings.simulate:
            self.last_model = "simulated"
            return simulate_mod.synthesize_local(question, chunks)
        candidates = [(self.settings.llm_endpoint, self.settings.llm_model)]
        fallback = (
            self.settings.llm_fallback_endpoint,
            self.settings.llm_fallback_model,
        )
        if fallback[0] and fallback not in candidates:
            candidates.append(fallback)
        answer, model = synthesize_mod.synthesize_candidates(
            question, chunks, candidates, self.settings.max_tokens
        )
        self.last_model = model
        return answer

    def _ensure_index(self) -> VectorIndex:
        """Build (once) and return the vector index for the configured root.

        Thread-safe: FastAPI runs sync endpoints on a thread pool, so two
        concurrent first queries may race into this method; the lock makes the
        expensive ingest + embed step run exactly once.
        """
        if self._index is not None:
            return self._index
        with self._index_lock:
            if self._index is not None:
                return self._index
            file_chunks = ingestion.ingest(self.settings.root)
            chunks = ingestion.to_chunks(file_chunks)
            if not chunks:
                self._chunks = []
                self._index = VectorIndex([], np.zeros((0, 0), dtype=np.float32))
                return self._index
            matrix = self._embedder([c.text for c in chunks])
            self._chunks = chunks
            self._index = VectorIndex(chunks, matrix)
            return self._index

    @property
    def index_size(self) -> int:
        """Number of chunks in the index (0 if not yet built)."""
        return self._index.size if self._index is not None else 0

    # -- querying --------------------------------------------------------

    def retrieve(self, question: str, k: int | None = None) -> list[Source]:
        """Retrieve the top-``k`` passages for ``question`` without answering.

        Args:
            question: The query text.
            k: How many passages (defaults to the configured top_k).

        Returns:
            Ordered :class:`Source` citations, most relevant first.
        """
        index = self._ensure_index()
        k = k or self.settings.top_k
        qvec = self._embedder([question])[0]
        pairs = index.query(qvec, k)
        return [self._to_source(pos, score) for pos, score in pairs]

    def _to_source(self, pos: int, score: float) -> Source:
        chunk = self._chunks[pos]
        lines = chunk.text.splitlines()
        snippet = "\n".join(lines[:_SNIPPET_LINES]).strip()
        if len(snippet) > _SNIPPET_CHARS:
            snippet = snippet[:_SNIPPET_CHARS].rstrip() + "…"
        return Source(
            chunk.path, chunk.symbol, chunk.start_line, chunk.end_line, score, snippet
        )

    def ask(self, question: str, k: int | None = None) -> RagResult:
        """Run the full RAG pipeline and return a grounded, cited answer.

        Args:
            question: The user's question.
            k: How many passages to retrieve (defaults to the configured top_k).

        Returns:
            A :class:`RagResult` with the answer, its source citations, the
            model/engine used, and the wall-clock latency in milliseconds.
        """
        started = time.perf_counter()
        sources = self.retrieve(question, k)
        if not self._chunks:
            # No Python source found (empty tree or nothing indexable).
            answer = "The codebase could not be indexed (no Python source found)."
        elif not sources:
            answer = "No relevant passages were found in the indexed codebase."
        else:
            answer = self._synthesizer(question, self._chunks_for(sources))
        return RagResult(
            question=question,
            answer=answer,
            sources=sources,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            model=self.last_model or self.settings.llm_model,
            engine="local-rag",
        )

    # -- helpers ---------------------------------------------------------

    def _chunks_for(self, sources: Sequence[Source]) -> list[Chunk]:
        """Map sources back to their full :class:`Chunk` text (by id)."""
        by_id = {c.id: c for c in self._chunks}
        out: list[Chunk] = []
        for s in sources:
            chunk = by_id.get(f"{s.path}::{s.symbol}:{s.start_line}")
            if chunk is not None:
                out.append(chunk)
        return out
