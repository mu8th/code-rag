"""Tests for the offline simulation engine (no endpoints, no models)."""

from __future__ import annotations

import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np

from coderag import CodeRAG
from coderag.config import Settings
from coderag.models import Chunk
from coderag.services.simulate import (
    EMBED_DIM,
    LexicalIndex,
    _stem,
    local_embed,
    synthesize_local,
)


def test_local_embed_is_deterministic() -> None:
    a1 = local_embed(["def foo():\n    return 1"])
    a2 = local_embed(["def foo():\n    return 1"])
    assert a1.shape == (1, EMBED_DIM)
    np.testing.assert_array_equal(a1, a2)


def test_local_embed_normalizes_rows() -> None:
    matrix = local_embed(["alpha beta gamma", "def retrieve_items(query): pass"])
    norms = np.linalg.norm(matrix, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_local_embed_empty_input() -> None:
    matrix = local_embed([])
    assert matrix.shape == (0, 0)


def test_local_embed_shares_signal_for_similar_text() -> None:
    query = local_embed(["how do I retrieve matching items"])[0]
    related = local_embed(["def retrieve_items(query):\n    return items"])[0]
    unrelated = local_embed(["def paint_house(wall):\n    wall.color = red"])[0]
    assert float(np.dot(query, related)) > float(np.dot(query, unrelated))


def test_synthesize_local_cites_top_chunk() -> None:
    chunks = [
        Chunk("pkg/alpha.py", "retrieve_items", 5, 7, "def retrieve_items(q):\n    return q"),
        Chunk("pkg/beta.py", "helper", 1, 2, "def helper():\n    return 42"),
    ]
    answer = synthesize_local("how do I retrieve items?", chunks)
    assert "`retrieve_items` in pkg/alpha.py (lines 5-7) [1]" in answer
    # No docstring to quote: the signature stands in, and [2] is listed.
    assert "def retrieve_items(q):" in answer
    assert "[2]" in answer


def test_synthesize_local_empty_chunks() -> None:
    assert "No relevant passages" in synthesize_local("q", [])


def _settings(root: Path, tmp: Path) -> Settings:
    return Settings(
        root=root,
        embed_endpoint="http://unused/api/embed",
        embed_model="fake-embed",
        llm_endpoint="http://unused/v1/chat/completions",
        llm_model="fake-llm",
        top_k=2,
        max_tokens=64,
        bind_host="127.0.0.1",
        bind_port=0,
        static_dir=tmp,
    )


def test_pipeline_runs_fully_offline_in_simulate_mode(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "alpha.py").write_text(
        textwrap.dedent(
            """
            def retrieve_items(query):
                return [q for q in QUERIES if query in q]
            """
        ),
        encoding="utf-8",
    )
    (root / "pkg" / "beta.py").write_text(
        textwrap.dedent(
            """
            def paint_house(wall):
                wall.color = red
            """
        ),
        encoding="utf-8",
    )
    # No embedder/synthesizer injected: the defaults must run offline.
    rag = CodeRAG(_settings(root, tmp_path))

    result = rag.ask("How do I retrieve matching items?")

    assert result.model == "simulated"
    assert result.sources, "simulation mode should still retrieve real passages"
    assert result.sources[0].symbol == "retrieve_items"
    assert "retrieve_items" in result.answer
    assert result.latency_ms >= 0


# -- extractive synthesis ------------------------------------------------------

_RANKER = textwrap.dedent(
    '''
    class Ranker:
        """Rank passages by cosine similarity to the query vector.

        The highest score comes first. Ties keep their index order.

        Args:
            k: How many passages to keep, highest score first.
        """
    '''
)


def test_synthesize_local_quotes_matching_docstring_sentences() -> None:
    chunks = [
        Chunk("pkg/rank.py", "Ranker", 1, 9, _RANKER),
        Chunk("pkg/other.py", "helper", 1, 2, "def helper():\n    return 42"),
    ]
    answer = synthesize_local("How are passages ranked by cosine score?", chunks)
    lead = "The closest match is `Ranker` in pkg/rank.py (lines 1-9) [1]."
    assert answer.startswith(lead)
    assert "Rank passages by cosine similarity to the query vector. [1]" in answer
    # The Args: header is dropped and the parameter label stripped.
    assert "Args:" not in answer and "k:" not in answer
    assert "See also [2] `helper` in pkg/other.py" in answer


def test_synthesize_local_skips_incidental_one_word_overlaps() -> None:
    text = '"""Build the cache once. Rank passages by cosine similarity score."""'
    answer = synthesize_local(
        "How does the system build citations for passages?",
        [Chunk("pkg/a.py", "<module>", 1, 1, text)],
    )
    # "Build the cache once." shares only "build" with a 4-term question.
    assert "Build the cache once." not in answer
    assert answer.startswith("The closest match is the `pkg/a.py` module [1].")


def test_stemming_joins_word_forms() -> None:
    assert {_stem(w) for w in ("citations", "citing", "cites")} == {"cit"}
    assert {_stem(w) for w in ("embedding", "embeds", "embedded")} == {"embed"}
    assert _stem("chunks") == _stem("chunk") == "chunk"


# -- BM25 lexical index ----------------------------------------------------------


def _chunks() -> list[Chunk]:
    return [
        Chunk("pkg/rank.py", "Ranker", 1, 9, _RANKER),
        Chunk("pkg/ingest.py", "split_files", 1, 3,
              'def split_files(root):\n    """Split each file into symbol chunks."""'),
        Chunk("pkg/io.py", "save", 1, 2, "def save(path, data):\n    path.write(data)"),
    ]


def test_lexical_index_ranks_the_explaining_passage_first() -> None:
    index = LexicalIndex(_chunks())
    (pos, score), *_ = index.search("how are passages ranked by cosine similarity", 3)
    assert pos == 0
    assert 0.0 < score <= 1.0


def test_lexical_index_scores_unrelated_questions_zero() -> None:
    index = LexicalIndex(_chunks())
    assert index.search("best pizza recipe with mozzarella", 1)[0][1] == 0.0


def test_pipeline_refuses_below_relevance_floor(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "rank.py").write_text(_RANKER, encoding="utf-8")
    settings = replace(_settings(root, tmp_path), min_score=0.2)
    rag = CodeRAG(settings)

    off = rag.ask("What is the capital of France?")
    assert not off.confident and off.sources == []
    assert off.answer.startswith("No confident match in this codebase")
    assert off.model == "simulated"

    on = rag.ask("How are passages ranked by cosine similarity?")
    assert on.confident and on.sources[0].symbol == "Ranker"
    assert on.best_score >= 0.2
    assert "class Ranker" in on.sources[0].text
    assert on.to_dict()["sources"][0]["text"].startswith("class Ranker")
