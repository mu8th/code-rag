"""Data models shared across the code-rag pipeline.

These are plain, framework-free dataclasses so the pipeline can be exercised
without FastAPI and so the same shapes can be serialized for the demo frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single indexable unit of source code.

    A chunk is one top-level symbol (function or class) from a file, or the
    module-level preamble (docstring, imports, top-level statements) when a file
    has code before its first definition.
    """

    path: str  # repo-relative file path
    symbol: str  # function/class name, or "<module>"
    start_line: int
    end_line: int
    text: str

    @property
    def id(self) -> str:
        """Stable identifier for a chunk."""
        return f"{self.path}::{self.symbol}:{self.start_line}"


@dataclass
class Source:
    """A retrieved passage returned to the synthesizer and the caller."""

    path: str
    symbol: str
    start_line: int
    end_line: int
    score: float
    snippet: str  # short excerpt shown to the user as the citation
    text: str = ""  # the full passage (capped), so a UI can expand a citation

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(self.score, 4),
            "snippet": self.snippet,
            "text": self.text,
        }


@dataclass
class RagResult:
    """The outcome of a single :meth:`CodeRAG.ask` call."""

    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""
    engine: str = ""
    #: False when the best passage fell below the relevance floor; the answer
    #: then says so and ``sources`` is empty.
    confident: bool = True
    #: Score of the best passage, reported even when it was not confident.
    best_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "latency_ms": self.latency_ms,
            "model": self.model,
            "engine": self.engine,
            "confident": self.confident,
            "best_score": round(self.best_score, 4),
        }
