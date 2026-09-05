"""Tests for the CodeRAG pipeline using fake embedder/synthesizer (no live model)."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from coderag import CodeRAG
from coderag.config import Settings
from coderag.models import Chunk


def _settings(root: Path, tmp: Path) -> Settings:
    return Settings(
        root=root,
        embed_endpoint="http://unused/api/embed",
        embed_model="fake-embed",
        llm_endpoint="http://unused/v1/chat/completions",
        llm_model="fake-llm",
        top_k=2,
        max_tokens=64,
        bind_host="127.0.0.1",
        bind_port=0,
        static_dir=tmp,
    )


def _make_repo(tmp: Path) -> Path:
    root = tmp / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "alpha.py").write_text(
        textwrap.dedent(
            '''
            """Alpha module."""
            import os


            def retrieve_items(query):
                """Return matching items for a query string."""
                return [q for q in QUERIES if query in q]


            class Store:
                """In-memory storage for items."""

                def put(self, key, value):
                    self.data[key] = value
            '''
        ),
        encoding="utf-8",
    )
    (root / "pkg" / "beta.py").write_text(
        textwrap.dedent(
            """
            def unrelated_helper():
                return 42
            """
        ),
        encoding="utf-8",
    )
    return root


class DeterministicEmbedder:
    """Maps text to a vector whose first coordinate encodes a keyword match.

    Both the query about retrieving items and the ``retrieve_items`` chunk map
    to the first axis, so cosine ranking puts that chunk first.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        vecs = []
        for t in texts:
            v = [0.0, 0.0, 0.0]
            if "retriev" in t:  # query "retrieving" and chunk "retrieve_items"
                v[0] = 1.0
            elif "unrelated_helper" in t:
                v[1] = 1.0
            else:
                v[2] = 1.0
            vecs.append(v)
        return np.asarray(vecs, dtype=np.float32)


def _fake_synthesizer(question: str, chunks: Sequence[Chunk]) -> str:
    return "ANSWER:" + ",".join(c.symbol for c in chunks)


def test_ask_retrieves_and_synthesizes(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    embedder = DeterministicEmbedder()
    rag = CodeRAG(
        _settings(root, tmp_path), embedder=embedder, synthesizer=_fake_synthesizer
    )

    result = rag.ask("How do I retrieve matching items?")
    # The synthesizer should be called with the retrieve_items chunk first.
    assert result.answer.startswith("ANSWER:retrieve_items")
    assert result.model == "fake-llm"
    assert result.engine == "local-rag"
    assert result.latency_ms >= 0
    # Sources carry real file/symbol info.
    top = result.sources[0]
    assert top.symbol == "retrieve_items"
    assert top.path == "pkg/alpha.py"
    assert top.start_line <= top.end_line
    assert top.snippet
    # top_k respected
    assert len(result.sources) <= 2


def test_index_is_built_once(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    embedder = DeterministicEmbedder()
    rag = CodeRAG(
        _settings(root, tmp_path), embedder=embedder, synthesizer=_fake_synthesizer
    )
    rag.ask("q1")
    calls_after_first = embedder.calls
    rag.ask("q2")
    # Index embeds only once; the second ask only embeds the query (1 call).
    assert embedder.calls == calls_after_first + 1
    assert rag.index_size > 0


def test_ask_empty_codebase(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    rag = CodeRAG(
        _settings(root, tmp_path),
        embedder=DeterministicEmbedder(),
        synthesizer=_fake_synthesizer,
    )
    result = rag.ask("anything")
    assert "could not be indexed" in result.answer
    assert result.sources == []
