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
    #: Fallback chat endpoint, tried only when the primary refuses connections
    #: (e.g. LM Studio not running but Ollama is). Empty disables the fallback;
    #: :func:`get_settings` defaults it to Ollama's OpenAI-compatible API so
    #: the demo works with a single local runtime.
    llm_fallback_endpoint: str = ""
    #: Model id for the fallback endpoint (used only when the fallback
    #: endpoint is set).
    llm_fallback_model: str = ""
    #: Run fully offline (the demo default): deterministic local embeddings and
    #: rule-based synthesis, so no model endpoints are contacted at all. Set to
    #: False (``RAG_SIMULATE=0``) to use the real local model endpoints.
    simulate: bool = True
    #: Relevance floor. When the best passage scores below it, the pipeline
    #: answers "no confident match" instead of citing weak passages.
    min_score: float = 0.0
    #: Leave test files (``tests/``, ``test_*.py``) out of the index, so
    #: answers cite the implementation rather than its tests.
    skip_tests: bool = False


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
        "venvs",
        ".idea",
        ".vscode",
    }
)


#: Default relevance floor for simulation mode's normalized BM25 score.
#: Calibrated on this repository: questions about the code score 0.3-0.9,
#: unrelated ones ("best pizza recipe", "Kubernetes ingress") 0.0-0.12.
SIMULATED_MIN_SCORE = 0.2


def _flag(name: str, default: str) -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def get_settings() -> Settings:
    """Build :class:`Settings` from the environment.

    Every field falls back to a local default; override via the ``RAG_*``
    environment variables (see :mod:`coderag.config`).
    """
    root = Path(os.environ.get("RAG_ROOT", str(_PROJECT_DIR)))
    simulate = _flag("RAG_SIMULATE", "1")
    default_floor = str(SIMULATED_MIN_SCORE) if simulate else "0"
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
        llm_fallback_endpoint=os.environ.get(
            "RAG_LLM_FALLBACK_ENDPOINT",
            "http://localhost:11434/v1/chat/completions",
        ),
        llm_fallback_model=os.environ.get("RAG_LLM_FALLBACK_MODEL", "qwen3:4b"),
        top_k=int(os.environ.get("RAG_TOP_K", "4")),
        max_tokens=int(os.environ.get("RAG_MAX_TOKENS", "512")),
        bind_host=os.environ.get("RAG_HOST", "127.0.0.1"),
        bind_port=int(os.environ.get("RAG_PORT", "8090")),
        static_dir=_PROJECT_DIR / "frontend",
        simulate=simulate,
        min_score=float(os.environ.get("RAG_MIN_SCORE", default_floor)),
        skip_tests=_flag("RAG_SKIP_TESTS", "0"),
    )
