"""Answer synthesis via a local OpenAI-compatible chat endpoint (LM Studio).

Builds a grounded prompt from the retrieved passages and asks the model to
answer using only that context, citing the source symbols. A JSON "no-reasoning"
hint is sent so reasoning models return the answer in ``content`` promptly.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from typing import Any, cast

from ..models import Chunk

#: System instruction that keeps answers grounded in the retrieved context.
SYSTEM_PROMPT = (
    "You are a precise code assistant. Answer the question using ONLY the code "
    "context provided. Cite the specific file, symbol, and line range you rely on. "
    "Be concise (2-5 sentences) and use Markdown code formatting where helpful. "
    "If the context is insufficient, say so plainly rather than guessing."
)


def _context_block(chunks: Sequence[Chunk]) -> str:
    """Format retrieved chunks into the numbered context block for the prompt."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] {c.path}::{c.symbol} (lines {c.start_line}-{c.end_line})\n{c.text}"
        )
    return "\n\n".join(parts)


def _post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
    """POST JSON to an OpenAI-compatible chat endpoint and return the body."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return cast("dict[str, object]", body)


def synthesize(
    question: str,
    chunks: Sequence[Chunk],
    endpoint: str,
    model: str,
    max_tokens: int = 512,
) -> str:
    """Ask the local LLM to answer ``question`` from the retrieved ``chunks``.

    Args:
        question: The user's question.
        chunks: Retrieved passages (ordered, most relevant first).
        endpoint: OpenAI-compatible ``/chat/completions`` URL.
        model: Model id.
        max_tokens: Generation budget.

    Returns:
        The assistant's answer text.

    Raises:
        RuntimeError: If the endpoint response has no usable content.
        TypeError: If the response message is not a JSON object.
    """
    context = _context_block(chunks)
    payload = {
        "model": model,
        "reasoning_effort": "none",
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Code context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ],
    }
    resp = _post(endpoint, payload)
    try:
        choices: Any = resp["choices"]
        message: Any = choices[0]["message"]
    except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Malformed LLM response: {exc}") from exc
    if not isinstance(message, dict):
        raise TypeError("Malformed LLM response: message is not an object")
    content = message.get("content") or ""
    if not str(content).strip():
        # Some reasoning models put the answer in reasoning_content when the
        # content field is empty or null; surface that rather than nothing.
        content = message.get("reasoning_content") or ""
    answer = str(content).strip()
    if not answer:
        raise RuntimeError("LLM returned an empty answer")
    return answer
