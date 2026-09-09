"""Embedding via a local Ollama endpoint.

Batch-embeds chunk text through Ollama's ``/api/embed``. The call is plain
``urllib`` (standard library) so the package has no hard dependency on a
specific HTTP client; the demo backend already ships one, but the core
pipeline stays dependency-light.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

#: How many texts to send per HTTP request.
_BATCH = 32


def _post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
    """POST ``payload`` (JSON) to ``endpoint`` and return the decoded response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return cast("dict[str, object]", body)


def embed_texts(texts: Sequence[str], endpoint: str, model: str) -> np.ndarray:
    """Embed ``texts`` and return an ``(N, dim)`` float32 matrix.

    Args:
        texts: Strings to embed (may be empty).
        endpoint: Ollama ``/api/embed`` URL.
        model: Embedding model name.

    Returns:
        A numpy matrix of shape ``(len(texts), dim)``. An empty input returns an
        empty ``(0, 0)`` array so callers can branch on shape.

    Raises:
        RuntimeError: If the endpoint returns no embeddings for a non-empty batch.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = list(texts[i : i + _BATCH])
        resp = _post(endpoint, {"model": model, "input": batch})
        emb: Any = resp.get("embeddings")
        if not emb or len(emb) != len(batch):
            raise RuntimeError(
                f"Embedding endpoint returned {len(emb or [])} vectors for {len(batch)} texts"
            )
        vectors.extend(emb)

    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix
