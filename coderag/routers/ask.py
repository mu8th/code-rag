"""Ask endpoint: run the full RAG pipeline for a question."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.pipeline import CodeRAG

router = APIRouter(prefix="/api", tags=["rag"])


class AskRequest(BaseModel):
    """A question to answer against the indexed codebase."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class AskResponse(BaseModel):
    """The grounded answer plus its citations and timing."""

    question: str
    answer: str
    sources: list[dict[str, object]]
    latency_ms: float
    model: str
    engine: str


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    """Answer ``body.question`` using the shared codebase index.

    The :class:`CodeRAG` instance is created once at startup and stored on
    ``app.state`` so the vector index is built a single time and reused.
    """
    rag: CodeRAG = request.app.state.rag
    try:
        result = rag.ask(body.question, k=body.top_k)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=502, detail=f"RAG pipeline failed: {exc}"
        ) from exc
    return AskResponse(**result.to_dict())
