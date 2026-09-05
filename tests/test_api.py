"""Tests for the FastAPI demo server endpoints (no live model required)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from fastapi.testclient import TestClient

from coderag import CodeRAG
from coderag.config import Settings
from coderag.main import app
from coderag.models import Chunk


class FakeEmbedder:
    def __init__(self) -> None:
        self.dim = 2

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i][0 if "retrieve_items" in t else 1] = 1.0
        return out


def _fake_synthesizer(question: str, chunks: Sequence[Chunk]) -> str:
    return "ok:" + ",".join(c.symbol for c in chunks)


def _build_client(tmp_path) -> TestClient:
    settings = Settings(
        root=tmp_path,
        embed_endpoint="http://unused",
        embed_model="fake",
        llm_endpoint="http://unused",
        llm_model="fake-llm",
        top_k=3,
        max_tokens=32,
        bind_host="127.0.0.1",
        bind_port=0,
        static_dir=tmp_path,
    )
    # Bypass the startup event: attach the shared RAG instance directly.
    app.state.rag = CodeRAG(
        settings, embedder=FakeEmbedder(), synthesizer=_fake_synthesizer
    )
    return TestClient(app)


def test_status_reports_ok(tmp_path) -> None:
    client = _build_client(tmp_path)
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["llm_model"] == "fake-llm"
    assert "index_size" in data


def test_ask_returns_answer_and_sources(tmp_path) -> None:
    # seed a tiny repo so retrieval finds the right symbol
    (tmp_path / "m.py").write_text(
        "def retrieve_items(query):\n    return []\n", encoding="utf-8"
    )
    client = _build_client(tmp_path)
    r = client.post("/api/ask", json={"question": "How do I retrieve matching items?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"].startswith("ok:")
    assert isinstance(data["sources"], list)
    assert data["sources"][0]["symbol"] == "retrieve_items"
    assert "latency_ms" in data


def test_ask_rejects_empty_question(tmp_path) -> None:
    client = _build_client(tmp_path)
    r = client.post("/api/ask", json={"question": ""})
    assert r.status_code == 422
