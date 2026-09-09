"""Runtime configuration for code-rag.

All values are read from environment variables with sensible local defaults so
the package works out of the box against a local Ollama embedding endpoint and
a local LM Studio OpenAI-compatible chat endpoint. Nothing here ever points at
a public host: the demo runs entirely on the developer's machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# code-rag/coderag  ->  code-rag
_PACKAGE_DIR: Path = Path(__file__).resolve().parent
_PROJECT_DIR: Path = _PACKAGE_DIR.parent


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for a :class:`~coderag.services.pipeline.CodeRAG`."""

    #: Codebase to index (defaults to the project itself, so the assistant can
    #: answer questions about its own source).
    root: Path
    #: Ollama embedding endpoint (``/api/embed``).
    embed_endpoint: str
    #: Embedding model name.
    embed_model: str
    #: OpenAI-compatible chat completions endpoint (LM Studio).
    llm_endpoint: str
    #: Chat model id.
    llm_model: str
    #: Number of retrieved passages passed to the synthesizer.
    top_k: int
    #: Generation budget for the synthesizer.
    max_tokens: int
    #: Network binding for the standalone demo server (localhost only).
    bind_host: str
    bind_port: int
    #: Directory containing the static demo frontend.
    static_dir: Path


#: Directories excluded from the repository walk.
WALK_SKIP: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
    }
)


def get_settings() -> Settings:
    """Build :class:`Settings` from the environment.

    Every field falls back to a local default; override via the ``RAG_*``
    environment variables (see :mod:`coderag.config`).
    """
    root = Path(os.environ.get("RAG_ROOT", str(_PROJECT_DIR)))
    return Settings(
        root=root,
        embed_endpoint=os.environ.get(
            "RAG_EMBED_ENDPOINT", "http://localhost:11434/api/embed"
        ),
        embed_model=os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text"),
        llm_endpoint=os.environ.get(
            "RAG_LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions"
        ),
        llm_model=os.environ.get("RAG_LLM_MODEL", "dirk-qwen3.8-27b@q4_k_s"),
        top_k=int(os.environ.get("RAG_TOP_K", "4")),
        max_tokens=int(os.environ.get("RAG_MAX_TOKENS", "512")),
        bind_host=os.environ.get("RAG_HOST", "127.0.0.1"),
        bind_port=int(os.environ.get("RAG_PORT", "8090")),
        static_dir=_PROJECT_DIR / "frontend",
    )
