"""Codebase ingestion: turn a tree of Python files into semantic chunks.

Each top-level function or class becomes one chunk so retrieval lands on the
right *symbol*, not an arbitrary line window. The module-level preamble
(docstring, imports, top-level statements) is kept as its own chunk so questions
about "what does this module import" still resolve.

Chunking is done with the standard-library :mod:`ast`, so it is robust to
formatting and requires no third-party parser.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..config import WALK_SKIP
from ..models import Chunk

#: File extensions considered for indexing.
CODE_EXTS = {".py"}

#: Maximum number of lines kept in a single chunk's ``text`` (guards the
#: embedding budget against a very large class body).
_MAX_CHUNK_LINES = 220


@dataclass
class FileChunk:
    """A file and the raw text of one of its chunks."""

    rel_path: str
    symbol: str
    start_line: int
    end_line: int
    text: str


def _iter_code_files(root: Path) -> Iterator[Path]:
    """Yield every indexable source file under ``root`` (skipping VCS/caches)."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in WALK_SKIP]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in CODE_EXTS:
                yield Path(dirpath) / name


def _preamble(source: str, tree: ast.Module) -> tuple[str, int] | None:
    """Return ``(preamble_text, end_line)`` for module-level non-definition code.

    The preamble covers everything at module level that is not a top-level
    function or class: the docstring, imports, and constants, wherever they
    appear in the file. It is built from the AST (not line matching), so files
    with only constants are still indexed and an indented ``def`` inside some
    other statement can never cut the preamble off early. Returns ``None`` when
    the file has nothing to keep.
    """
    lines = source.splitlines()
    segments: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", None) or start
        segments.append((start, end))
    if not segments:
        return None
    text = "\n\n".join("\n".join(lines[s - 1 : e]) for s, e in segments).strip()
    if not text:
        return None
    return (text, max(e for _, e in segments))


def _chunk_functions(source: str, rel_path: str) -> list[FileChunk]:
    """Split one file into per-symbol chunks using the AST."""
    tree = ast.parse(source)
    lines = source.splitlines()
    out: list[FileChunk] = []

    preamble = _preamble(source, tree)
    if preamble:
        text, end_line = preamble
        out.append(FileChunk(rel_path, "<module>", 1, end_line, text))

    for node in tree.body:
        name = getattr(node, "name", None)
        if name is None:
            continue  # skip bare statements at module level (already in preamble)
        start = node.lineno
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            # Include the decorator lines so they are not lost between the
            # preamble and the definition itself.
            start = min(d.lineno for d in decorators)
        end = getattr(node, "end_lineno", None) or node.lineno
        # Cap very long bodies so a single huge class does not dominate the
        # embedding budget; the first N lines still carry the signature + body.
        kept = min(end - start + 1, _MAX_CHUNK_LINES)
        text = "\n".join(lines[start - 1 : start - 1 + kept])
        out.append(FileChunk(rel_path, name, start, min(end, start + kept - 1), text))
    return out


def ingest(root: Path) -> list[FileChunk]:
    """Index ``root`` into a flat list of :class:`FileChunk`.

    Args:
        root: Directory to walk.

    Returns:
        One :class:`FileChunk` per top-level symbol (plus module preambles).
        Files that fail to parse are skipped rather than raising, so one
        malformed file never breaks the whole index.
    """
    chunks: list[FileChunk] = []
    for path in _iter_code_files(root):
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        try:
            file_chunks = _chunk_functions(source, rel)
        except (SyntaxError, ValueError):
            # Fall back to one coarse chunk so the file is still searchable.
            text = source[: _MAX_CHUNK_LINES * 80]
            if text.strip():
                file_chunks = [
                    FileChunk(rel, "<unparsed>", 1, source.count("\n") + 1, text)
                ]
        chunks.extend(file_chunks)
    return chunks


def to_chunks(file_chunks: list[FileChunk]) -> list[Chunk]:
    """Convert :class:`FileChunk` records into :class:`~coderag.models.Chunk`."""
    return [
        Chunk(fc.rel_path, fc.symbol, fc.start_line, fc.end_line, fc.text)
        for fc in file_chunks
    ]
