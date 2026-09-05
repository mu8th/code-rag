"""Tests for the numpy cosine retrieval index."""

from __future__ import annotations

import numpy as np
import pytest

from coderag.models import Chunk
from coderag.services.retrieval import VectorIndex


def _chunks() -> list[Chunk]:
    return [
        Chunk("a.py", "f", 1, 2, "def f(): pass"),
        Chunk("b.py", "g", 1, 2, "def g(): pass"),
    ]


def test_query_ranks_by_similarity() -> None:
    # chunk 0 is aligned with the query, chunk 1 is orthogonal
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = VectorIndex(_chunks(), matrix)
    pairs = index.query(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert pairs[0][0] == 0
    assert pairs[0][1] == pytest.approx(1.0, abs=1e-5)
    assert pairs[1][0] == 1
    assert pairs[1][1] == pytest.approx(0.0, abs=1e-5)


def test_query_k_limits_results() -> None:
    # Three distinct 2-D directions; the query aligns most with chunk 0.
    matrix = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    index = VectorIndex([Chunk("a.py", f"f{i}", 1, 1, "") for i in range(3)], matrix)
    pairs = index.query(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert len(pairs) == 2
    assert pairs[0][0] == 0  # chunk 0 is the most aligned


def test_query_empty_index() -> None:
    index = VectorIndex([], np.zeros((0, 0), dtype=np.float32))
    assert index.query(np.array([1.0, 0.0], dtype=np.float32), k=3) == []


def test_chunk_count_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        VectorIndex(_chunks(), np.ones((3, 2), dtype=np.float32))


def test_chunks_at_roundtrip() -> None:
    matrix = np.array([[1.0], [0.5]], dtype=np.float32)
    chunks = _chunks()
    index = VectorIndex(chunks, matrix)
    got = index.chunks_at([0, 1])
    assert got == chunks
