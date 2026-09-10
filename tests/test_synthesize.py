"""Tests for the grounded answer synthesizer (prompt + response handling)."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest import mock

import pytest

from coderag.models import Chunk
from coderag.services.synthesize import synthesize, synthesize_candidates


def _fake_resp(content: str) -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "",
                    }
                }
            ]
        }
    ).encode("utf-8")


def test_synthesize_returns_content_and_sends_no_reasoning() -> None:
    chunks = [Chunk("a.py", "f", 1, 3, "def f():\n    return 1\n")]
    captured: list[dict[str, Any]] = []

    def fake_urlopen(req, timeout=None):
        captured.append(json.loads(req.data.decode("utf-8")))
        return io.BytesIO(_fake_resp("f returns 1."))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = synthesize(
            "what does f do?", chunks, "http://x/chat", "model", max_tokens=128
        )

    assert out == "f returns 1."
    payload = captured[0]
    assert payload["reasoning_effort"] == "none"
    assert payload["max_tokens"] == 128
    # context block is present in the user message
    assert "a.py::f" in payload["messages"][1]["content"]
    assert "def f():" in payload["messages"][1]["content"]


def test_synthesize_falls_back_to_reasoning_content() -> None:
    resp = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "from reasoning",
                    }
                }
            ]
        }
    ).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(resp)

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = synthesize("q", [Chunk("a.py", "f", 1, 1, "x")], "http://x/chat", "m")
    assert out == "from reasoning"


def test_synthesize_empty_raises() -> None:
    def fake_urlopen(req, timeout=None):
        return io.BytesIO(_fake_resp("   "))

    with (
        mock.patch("urllib.request.urlopen", side_effect=fake_urlopen),
        pytest.raises(RuntimeError),
    ):
        synthesize("q", [Chunk("a.py", "f", 1, 1, "x")], "http://x/chat", "m")


def test_synthesize_null_content_falls_back_to_reasoning() -> None:
    # Reasoning models often send content: null; that must not crash.
    resp = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "the answer",
                    }
                }
            ]
        }
    ).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(resp)

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        out = synthesize("q", [Chunk("a.py", "f", 1, 1, "x")], "http://x/chat", "m")
    assert out == "the answer"


def test_synthesize_non_dict_message_raises_type_error() -> None:
    resp = json.dumps({"choices": [{"message": None}]}).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(resp)

    with (
        mock.patch("urllib.request.urlopen", side_effect=fake_urlopen),
        pytest.raises(TypeError),
    ):
        synthesize("q", [Chunk("a.py", "f", 1, 1, "x")], "http://x/chat", "m")


def test_candidates_fall_back_when_primary_refuses() -> None:
    chunks = [Chunk("a.py", "f", 1, 3, "def f():\n    return 1\n")]

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        if url == "http://primary/chat":
            raise urllib.error.URLError("connection refused")
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "fallback-model"
        return io.BytesIO(_fake_resp("answered by fallback."))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        answer, model = synthesize_candidates(
            "q",
            chunks,
            [
                ("http://primary/chat", "primary-model"),
                ("http://fallback/chat", "fallback-model"),
            ],
        )

    assert answer == "answered by fallback."
    assert model == "fallback-model"


def test_candidates_use_primary_when_healthy() -> None:
    chunks = [Chunk("a.py", "f", 1, 3, "def f():\n    return 1\n")]
    hits: list[str] = []

    def fake_urlopen(req, timeout=None):
        hits.append(req.full_url)
        return io.BytesIO(_fake_resp("primary answer."))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        answer, model = synthesize_candidates(
            "q",
            chunks,
            [
                ("http://primary/chat", "primary-model"),
                ("http://fallback/chat", "fallback-model"),
            ],
        )

    assert answer == "primary answer."
    assert model == "primary-model"
    assert hits == ["http://primary/chat"]  # fallback never contacted


def test_candidates_all_unreachable_raises() -> None:
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    with (
        mock.patch("urllib.request.urlopen", side_effect=fake_urlopen),
        pytest.raises(RuntimeError, match="All LLM endpoints unreachable"),
    ):
        synthesize_candidates(
            "q",
            [Chunk("a.py", "f", 1, 1, "x")],
            [
                ("http://primary/chat", "m1"),
                ("http://fallback/chat", "m2"),
            ],
        )


def test_candidates_no_endpoints_raises() -> None:
    with pytest.raises(RuntimeError, match="No LLM endpoints configured"):
        synthesize_candidates("q", [Chunk("a.py", "f", 1, 1, "x")], [])
