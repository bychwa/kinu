from __future__ import annotations

import pytest

from kinu.embed import embed_one
from kinu.engine import Engine, NamespaceNotFound
from kinu.filters import matches
from kinu.store.local import LocalStore


@pytest.fixture
def engine(tmp_path):
    return Engine(LocalStore(tmp_path))


def seed(engine: Engine, namespace: str = "t") -> None:
    engine.write(
        namespace,
        [
            {"id": "a", "vector": [1.0, 0.0], "text": "storage cost", "tenant": "acme", "page": 1},
            {"id": "b", "vector": [0.0, 1.0], "text": "pto policy", "tenant": "acme", "page": 2},
            {
                "id": "c",
                "vector": [0.9, 0.1],
                "text": "object storage",
                "tenant": "other",
                "page": 3,
            },
        ],
        distance_metric="cosine_distance",
    )


def test_query_returns_nearest_first(engine):
    seed(engine)
    result = engine.query("t", rank_by=["vector", "ANN", [1.0, 0.0]], limit=3, include=["text"])

    assert [row["id"] for row in result["rows"]] == ["a", "c", "b"]
    assert result["rows"][0]["$dist"] == pytest.approx(0.0, abs=1e-6)


def test_include_controls_payload(engine):
    seed(engine)
    narrow = engine.query("t", rank_by=["vector", "ANN", [1.0, 0.0]], limit=1, include=["page"])
    wide = engine.query(
        "t", rank_by=["vector", "ANN", [1.0, 0.0]], limit=1, include=["page", "text"]
    )

    assert "text" not in narrow["rows"][0]
    assert wide["rows"][0]["text"] == "storage cost"
    assert wide["performance"]["bytes_fetched"] > narrow["performance"]["bytes_fetched"]


def test_filtering(engine):
    seed(engine)
    result = engine.query(
        "t",
        rank_by=["vector", "ANN", [1.0, 0.0]],
        filters=["And", [["tenant", "Eq", "acme"], ["page", "Lt", 2]]],
        limit=10,
    )

    assert [row["id"] for row in result["rows"]] == ["a"]


def test_delete_hides_the_row(engine):
    seed(engine)
    engine.delete("t", ["a"])

    result = engine.query("t", rank_by=["vector", "ANN", [1.0, 0.0]], limit=3)
    assert "a" not in [row["id"] for row in result["rows"]]


def test_unknown_namespace(engine):
    with pytest.raises(NamespaceNotFound):
        engine.query("nope", rank_by=["vector", "ANN", [1.0, 0.0]])


def test_dimension_mismatch_is_rejected(engine):
    seed(engine)
    with pytest.raises(ValueError):
        engine.write("t", [{"id": "z", "vector": [1.0, 2.0, 3.0]}])


def test_filter_operators():
    row = {"tenant": "acme", "page": 3, "tags": ["a", "b"]}

    assert matches(row, ["tenant", "Eq", "acme"])
    assert not matches(row, ["tenant", "Eq", "other"])
    assert matches(row, ["page", "Gte", 3])
    assert matches(row, ["tags", "ContainsAny", ["b", "z"]])
    assert matches(row, ["Or", [["page", "Lt", 1], ["tenant", "Eq", "acme"]]])
    assert matches(row, ["Not", ["tenant", "Eq", "other"]])
    with pytest.raises(ValueError):
        matches(row, ["page", "Wat", 1])


def test_demo_corpus_shape():
    vector = embed_one("how much do we spend on storage")
    assert len(vector) == 1024
    assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-5


def test_embed_query_path(engine):
    """The ['Embed', text] ranking form embeds server-side, with namespace IDF."""
    from kinu.embed import embed, idf_weights, tokenize

    texts = ["storage cost breakdown", "pto policy details", "object storage notes"]
    document_freq: dict[str, int] = {}
    for text in texts:
        for token in set(tokenize(text)):
            document_freq[token] = document_freq.get(token, 0) + 1

    vectors = embed(texts, dim=512, idf=idf_weights(document_freq, len(texts)))
    engine.write(
        "e",
        [{"id": i, "vector": vectors[i].tolist(), "text": texts[i]} for i in range(len(texts))],
    )
    engine.put_stats("e", len(texts), document_freq)

    result = engine.query("e", rank_by=["vector", "ANN", ["Embed", "storage"]], limit=3)

    assert result["rows"][0]["id"] in (0, 2), "rows mentioning 'storage' should lead"
