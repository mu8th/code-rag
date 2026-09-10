"""Status endpoint: report index readiness and the configured models."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..config import Settings

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status")
def status(request: Request) -> dict[str, object]:
    """Return index size, the indexed root, and the configured model settings.

    This is a read-only view of the configuration; it does not probe the model
    endpoints. A query to ``/api/ask`` is the real reachability test, and it
    degrades with a clear 502 when a local model is offline.
    """
    rag = request.app.state.rag
    settings: Settings = rag.settings
    return {
        "status": "ok",
        "index_size": rag.index_size,
        "root": str(settings.root),
        "simulate": settings.simulate,
        "embed_model": settings.embed_model,
        "llm_model": settings.llm_model,
        "top_k": settings.top_k,
    }
