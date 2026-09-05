"""code-rag demo server.

A small FastAPI app that exposes the local RAG pipeline over HTTP and serves a
single-page demo frontend. It binds to ``127.0.0.1:8090`` by design (local-only,
never publicly hosted) and points its models at the developer's local Ollama
(embeddings) and LM Studio (chat) endpoints.

Run it with::

    uvicorn coderag.main:app --host 127.0.0.1 --port 8090

or ``python -m coderag.main``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .routers import ask as ask_router
from .routers import status as status_router
from .services.pipeline import CodeRAG

logger = logging.getLogger("code-rag")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the shared :class:`CodeRAG` once per app lifecycle.

    The index is built lazily on the first query, so startup stays fast even
    when the local models are not yet reachable.
    """
    settings = config.get_settings()
    app.state.rag = CodeRAG(settings)
    if settings.static_dir.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(settings.static_dir), html=True),
            name="frontend",
        )
    yield


app = FastAPI(title="code-rag", version="1.0.0", lifespan=lifespan)

# Allow the portfolio site (served on :8080) to call this API for its demo card.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


app.include_router(status_router.router)
app.include_router(ask_router.router)


if __name__ == "__main__":
    import uvicorn

    settings = config.get_settings()
    uvicorn.run(app, host=settings.bind_host, port=settings.bind_port, log_level="info")
