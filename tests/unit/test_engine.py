from __future__ import annotations

import pytest

from kinu.engine import Engine, NamespaceNotFound

QUERY = ["vector", "ANN", [1.0, 0.0]]


def test_query_returns_nearest_first(seeded: Engine):
    result = seeded.query("t", rank_by=QUERY, limit=3, include=["text"])

    assert [row["id"] for row in result["rows"]] == ["a", "c", "b"]
    assert result["rows"][0]["$dist"] == pytest.approx(0.0, abs=1e-6)


def test_include_controls_payload(seeded: Engine):
    narrow = seeded.query("t", rank_by=QUERY, limit=1, include=["page"])
    wide = seeded.query("t", rank_by=QUERY, limit=1, include=["page", "text"])

    assert "text" not in narrow["rows"][0]
    assert wide["rows"][0]["text"] == "storage cost"
    assert wide["performance"]["bytes_fetched"] > narrow["performance"]["bytes_fetched"]


def test_filtering(seeded: Engine):
    result = seeded.query(
        "t",
        rank_by=QUERY,
        filters=["And", [["tenant", "Eq", "acme"], ["page", "Lt", 2]]],
        limit=10,
    )

    assert [row["id"] for row in result["rows"]] == ["a"]


def test_delete_hides_the_row(seeded: Engine):
    seeded.delete("t", ["a"])

    result = seeded.query("t", rank_by=QUERY, limit=3)
    assert "a" not in [row["id"] for row in result["rows"]]


def test_unknown_namespace(engine: Engine):
    with pytest.raises(NamespaceNotFound):
        engine.query("nope", rank_by=QUERY)


def test_dimension_mismatch_is_rejected(seeded: Engine):
    with pytest.raises(ValueError):
        seeded.write("t", [{"id": "z", "vector": [1.0, 2.0, 3.0]}])


def test_bm25_rank_by_is_not_implemented_yet(seeded: Engine):
    with pytest.raises(NotImplementedError):
        seeded.query("t", rank_by=["text", "BM25", "storage"])


def test_embed_query_path(seeded: Engine):
    """The ['Embed', text] ranking form embeds server-side, with namespace IDF."""
    from kinu.embed import embed, idf_weights, tokenize

    texts = ["storage cost breakdown", "pto policy details", "object storage notes"]
    document_freq: dict[str, int] = {}
    for text in texts:
        for token in set(tokenize(text)):
            document_freq[token] = document_freq.get(token, 0) + 1

    vectors = embed(texts, dim=512, idf=idf_weights(document_freq, len(texts)))
    seeded.write(
        "e",
        [{"id": i, "vector": vectors[i].tolist(), "text": texts[i]} for i in range(len(texts))],
    )
    seeded.put_stats("e", len(texts), document_freq)

    result = seeded.query("e", rank_by=["vector", "ANN", ["Embed", "storage"]], limit=3)

    assert result["rows"][0]["id"] in (0, 2), "rows mentioning 'storage' should lead"
