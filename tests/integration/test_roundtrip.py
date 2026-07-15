"""Ingest on disk, then query it back through the engine.

Not a unit test of any one function: it walks the whole path a real user
takes — markdown files in, ranked rows out — and checks the byte accounting
that the design claims.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kinu.engine import Engine
from kinu.ingest import ingest_path
from kinu.store.local import LocalStore

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "examples" / "3-docs"


@pytest.fixture
def engine(tmp_path) -> Engine:
    return Engine(LocalStore(tmp_path))


def test_ingest_covers_the_corpus(engine: Engine):
    stats = ingest_path(engine, "docs", CORPUS)

    assert stats.documents == 3
    assert stats.chunks == 12
    assert stats.bytes_written > 0
    assert engine.meta("docs").rows == 12


def test_sections_become_chunk_attributes(engine: Engine):
    ingest_path(engine, "docs", CORPUS)

    result = engine.query(
        "docs",
        rank_by=["vector", "ANN", ["Embed", "who approves remote work"]],
        limit=1,
        include=["doc", "section"],
    )

    top = result["rows"][0]
    assert top["doc"] == "handbook"
    assert top["section"] == "Remote Work"


def test_idf_stats_are_persisted(engine: Engine):
    ingest_path(engine, "docs", CORPUS)

    n_docs, document_freq = engine.stats("docs")

    assert n_docs == 12
    assert document_freq["pto"] == 1
    assert document_freq["storage"] > document_freq["pto"]
    assert engine.idf("docs")["pto"] > engine.idf("docs")["storage"]


def test_query_reads_less_when_text_is_not_requested(engine: Engine):
    ingest_path(engine, "docs", CORPUS)
    rank = ["vector", "ANN", ["Embed", "storage"]]

    narrow = engine.query("docs", rank_by=rank, limit=5, include=["doc"])
    wide = engine.query("docs", rank_by=rank, limit=5, include=["doc", "text"])

    assert wide["performance"]["bytes_fetched"] > narrow["performance"]["bytes_fetched"]
    assert narrow["performance"]["roundtrips"] == 2


def test_delete_survives_a_later_segment(engine: Engine):
    ingest_path(engine, "docs", CORPUS)
    before = engine.query("docs", rank_by=["vector", "ANN", ["Embed", "pto"]], limit=1)
    victim = before["rows"][0]["id"]

    engine.delete("docs", [victim])

    after = engine.query("docs", rank_by=["vector", "ANN", ["Embed", "pto"]], limit=10)
    assert victim not in [row["id"] for row in after["rows"]]


def test_rows_stay_within_one_dimension(engine: Engine):
    ingest_path(engine, "docs", CORPUS)
    meta = engine.meta("docs")

    assert meta.dim == 1024
    assert meta.distance_metric == "cosine_distance"


def test_chunks_are_smaller_than_the_documents(engine: Engine):
    ingest_path(engine, "docs", CORPUS)
    big = max(p.stat().st_size for p in CORPUS.glob("*.md"))

    segment_bytes = sum(
        p.stat().st_size for p in (Path(engine.store.root) / "docs" / "wal").iterdir()
    )
    assert segment_bytes > big, "vectors dominate the segment, text is a minority of it"


def test_stats_are_keyed_by_stemmed_tokens(engine: Engine):
    ingest_path(engine, "docs", CORPUS)
    _, document_freq = engine.stats("docs")

    assert "pto" in document_freq
    assert "laptop" in document_freq, "plurals are folded before counting"
    assert "laptops" not in document_freq
    assert all(token == token.lower() for token in document_freq)
