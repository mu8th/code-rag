"""code-rag -- a local, dependency-light RAG code assistant.

Indexes a Python codebase into semantic chunks, embeds them with a local
embedding model, retrieves the most relevant passages, and synthesizes a
grounded, source-cited answer with a local LLM.

The public entry point is :class:`coderag.services.pipeline.CodeRAG`.
"""

from .models import Chunk, RagResult, Source
from .services.pipeline import CodeRAG

__all__ = ["Chunk", "CodeRAG", "RagResult", "Source"]
