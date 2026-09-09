"""Tests for AST-based codebase ingestion."""

from __future__ import annotations

import textwrap
from pathlib import Path

from coderag.services import ingestion


def _write(root: Path, rel: str, code: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(code), encoding="utf-8")


def test_ingest_splits_by_symbol(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        '''
        """Top docstring."""
        import os

        CONST = 1

        def alpha(x):
            return x + 1


        class Beta:
            def method(self):
                return 0
        ''',
    )
    chunks = ingestion.ingest(tmp_path)
    symbols = {c.symbol for c in chunks}
    # module preamble + alpha + Beta
    assert "<module>" in symbols
    assert "alpha" in symbols
    assert "Beta" in symbols
    # alpha chunk carries its body
    alpha = next(c for c in chunks if c.symbol == "alpha")
    assert "return x + 1" in alpha.text
    assert alpha.start_line <= alpha.end_line


def test_ingest_skips_cache_and_vcs_dirs(tmp_path: Path) -> None:
    _write(tmp_path, "keep.py", "def a():\n    return 1\n")
    _write(tmp_path, "__pycache__/junk.py", "def b():\n    return 2\n")
    _write(tmp_path, ".git/x.py", "def c():\n    return 3\n")
    chunks = ingestion.ingest(tmp_path)
    paths = {c.rel_path for c in chunks}
    assert "keep.py" in paths
    assert not any("__pycache__" in p for p in paths)
    assert not any(p.startswith(".git") for p in paths)


def test_ingest_skips_unparseable_file_with_fallback(tmp_path: Path) -> None:
    _write(tmp_path, "bad.py", "def broken(:\n   return 1\n")
    _write(tmp_path, "good.py", "def ok():\n    return 1\n")
    chunks = ingestion.ingest(tmp_path)
    # good.py is indexed normally
    assert any(c.symbol == "ok" for c in chunks)
    # bad.py is still represented via the unparsed fallback, not dropped
    assert any(c.rel_path == "bad.py" for c in chunks)


def test_ingest_empty_root(tmp_path: Path) -> None:
    assert ingestion.ingest(tmp_path) == []


def test_ingest_keeps_constants_only_file(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "consts.py",
        '''
        """Only constants here."""
        import os

        MAX_RETRIES = 3
        NAMES = ["a", "b"]
        ''',
    )
    chunks = ingestion.ingest(tmp_path)
    # A file with no top-level def/class must still be indexed via its preamble.
    assert any(c.rel_path == "consts.py" and c.symbol == "<module>" for c in chunks)
    module = next(c for c in chunks if c.symbol == "<module>")
    assert "MAX_RETRIES = 3" in module.text


def test_ingest_decorator_lines_stay_with_symbol(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "dec.py",
        '''
        import functools

        @functools.lru_cache(maxsize=None)
        def cached(x):
            return x * 2
        ''',
    )
    chunks = ingestion.ingest(tmp_path)
    cached = next(c for c in chunks if c.symbol == "cached")
    assert "@functools.lru_cache" in cached.text


def test_to_chunks_preserves_fields(tmp_path: Path) -> None:
    fc = [ingestion.FileChunk("a.py", "f", 1, 3, "def f():\n    pass\n")]
    chunks = ingestion.to_chunks(fc)
    assert chunks[0].path == "a.py"
    assert chunks[0].id == "a.py::f:1"
