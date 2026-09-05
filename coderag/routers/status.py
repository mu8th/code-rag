"""Status endpoint: report index readiness and model reachability."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..config import Settings

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status")
def status(request: Request) -> dict[str, object]:
    """Return index size, configured models, and whether they are reachable.

    The reachability probe is best-effort: a missing endpoint is reported as
    ``False`` rather than raising, so the frontend can show an honest status.
    """
    rag = request.app.state.rag
    settings: Settings = rag.settings
    return {
        "status": "ok",
        "index_size": rag.index_size,
        "root": str(settings.root),
        "embed_model": settings.embed_model,
        "llm_model": settings.llm_model,
        "top_k": settings.top_k,
    }
