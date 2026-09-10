"""Tests for the offline simulation engine (no endpoints, no models)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from coderag import CodeRAG
from coderag.config import Settings
from coderag.models import Chunk
from coderag.services.simulate import EMBED_DIM, local_embed, synthesize_local


def test_local_embed_is_deterministic() -> None:
    a1 = local_embed(["def foo():\n    return 1"])
    a2 = local_embed(["def foo():\n    return 1"])
    assert a1.shape == (1, EMBED_DIM)
    np.testing.assert_array_equal(a1, a2)


def test_local_embed_normalizes_rows() -> None:
    matrix = local_embed(["alpha beta gamma", "def retrieve_items(query): pass"])
    norms = np.linalg.norm(matrix, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_local_embed_empty_input() -> None:
    matrix = local_embed([])
    assert matrix.shape == (0, 0)


def test_local_embed_shares_signal_for_similar_text() -> None:
    query = local_embed(["how do I retrieve matching items"])[0]
    related = local_embed(["def retrieve_items(query):\n    return items"])[0]
    unrelated = local_embed(["def paint_house(wall):\n    wall.color = red"])[0]
    assert float(np.dot(query, related)) > float(np.dot(query, unrelated))


def test_synthesize_local_cites_top_chunk() -> None:
    chunks = [
        Chunk("pkg/alpha.py", "retrieve_items", 5, 7, "def retrieve_items(q):\n    return q"),
        Chunk("pkg/beta.py", "helper", 1, 2, "def helper():\n    return 42"),
    ]
    answer = synthesize_local("how do I retrieve items?", chunks)
    assert "pkg/alpha.py::retrieve_items" in answer
    assert "(lines 5-7)" in answer
    assert "[1]" in answer and "[2]" in answer


def test_synthesize_local_empty_chunks() -> None:
    assert "No relevant passages" in synthesize_local("q", [])


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


def test_pipeline_runs_fully_offline_in_simulate_mode(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "alpha.py").write_text(
        textwrap.dedent(
            """
            def retrieve_items(query):
                return [q for q in QUERIES if query in q]
            """
        ),
        encoding="utf-8",
    )
    (root / "pkg" / "beta.py").write_text(
        textwrap.dedent(
            """
            def paint_house(wall):
                wall.color = red
            """
        ),
        encoding="utf-8",
    )
    # No embedder/synthesizer injected: the defaults must run offline.
    rag = CodeRAG(_settings(root, tmp_path))

    result = rag.ask("How do I retrieve matching items?")

    assert result.model == "simulated"
    assert result.sources, "simulation mode should still retrieve real passages"
    assert result.sources[0].symbol == "retrieve_items"
    assert "retrieve_items" in result.answer
    assert result.latency_ms >= 0
