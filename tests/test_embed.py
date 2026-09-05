"""Tests for Ollama embedding (batching, empty input, error path)."""

from __future__ import annotations

import io
import json
from typing import Any
from unittest import mock

import pytest

from coderag.services import embed


def _fake_resp(vectors: list[list[float]]) -> bytes:
    return json.dumps({"embeddings": vectors}).encode("utf-8")


def test_embed_empty_returns_zero_shape() -> None:
    out = embed.embed_texts([], "http://x/api/embed", "m")
    assert out.shape == (0, 0)


def test_embed_batches_and_concatenates() -> None:
    # 33 texts -> two batches of 32 + 1 (batch size is 32). Each vector is a
    # distinct 2-d row so the concatenated matrix is homogeneous.
    texts = [f"t{i}" for i in range(33)]
    captured: list[dict[str, Any]] = []

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        captured.append(payload)
        n = len(payload["input"])
        vectors = [[float(i), float(i) * 2.0] for i in range(n)]
        return io.BytesIO(_fake_resp(vectors))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = embed.embed_texts(texts, "http://x/api/embed", "m")

    assert out.shape == (33, 2)
    assert len(captured) == 2
    assert len(captured[0]["input"]) == 32
    assert len(captured[1]["input"]) == 1


def test_embed_mismatched_count_raises() -> None:
    def fake_urlopen(req, timeout=None):
        return io.BytesIO(_fake_resp([[1.0, 2.0]]))  # only 1 vector for 2 texts

    with (
        mock.patch("urllib.request.urlopen", side_effect=fake_urlopen),
        pytest.raises(RuntimeError),
    ):
        embed.embed_texts(["a", "b"], "http://x/api/embed", "m")
