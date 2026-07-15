from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kinu import api

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def fresh_store(tmp_path, monkeypatch):
    from kinu.store.local import LocalStore

    monkeypatch.setattr(api, "engine", api.Engine(LocalStore(tmp_path)))
    yield


def test_write_then_query():
    write = client.post(
        "/v2/namespaces/docs",
        json={
            "upsert_rows": [
                {"id": 1, "vector": [1.0, 0.0], "text": "storage cost", "page": 1},
                {"id": 2, "vector": [0.0, 1.0], "text": "pto policy", "page": 2},
            ],
            "distance_metric": "cosine_distance",
        },
    )
    assert write.status_code == 200
    assert write.json()["rows_affected"] == 2

    response = client.post(
        "/v2/namespaces/docs/query",
        json={
            "rank_by": ["vector", "ANN", [1.0, 0.0]],
            "limit": 1,
            "include_attributes": ["text", "page"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows"][0]["id"] == 1
    assert body["rows"][0]["text"] == "storage cost"
    assert body["performance"]["bytes_fetched"] > 0
    assert body["billing"]["billable_logical_bytes_queried"] == body["performance"]["bytes_fetched"]


def test_missing_namespace_is_404():
    response = client.post(
        "/v2/namespaces/ghost/query", json={"rank_by": ["vector", "ANN", [1.0, 0.0]]}
    )
    assert response.status_code == 404


def test_metadata_and_delete():
    client.post(
        "/v2/namespaces/docs",
        json={
            "upsert_rows": [{"id": 1, "vector": [1.0, 0.0]}],
            "distance_metric": "cosine_distance",
        },
    )
    meta = client.get("/v2/namespaces/docs/metadata").json()
    assert meta["approx_row_count"] == 1

    assert client.delete("/v2/namespaces/docs").status_code == 200
    assert client.get("/v2/namespaces").json()["namespaces"] == []


def test_bad_rank_by_is_400():
    client.post(
        "/v2/namespaces/docs",
        json={
            "upsert_rows": [{"id": 1, "vector": [1.0, 0.0]}],
            "distance_metric": "cosine_distance",
        },
    )
    response = client.post("/v2/namespaces/docs/query", json={"rank_by": ["vector", "BM25", "x"]})
    assert response.status_code == 400


def test_health_and_namespace_listing():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v2/namespaces").json() == {"namespaces": []}

    client.post("/v2/namespaces/docs", json={"upsert_rows": [{"id": 1, "vector": [1.0, 0.0]}]})

    assert client.get("/v2/namespaces").json() == {"namespaces": ["docs"]}


def test_deletes_via_api():
    client.post(
        "/v2/namespaces/docs",
        json={
            "upsert_rows": [{"id": "a", "vector": [1.0, 0.0]}, {"id": "b", "vector": [0.0, 1.0]}],
            "distance_metric": "cosine_distance",
        },
    )

    response = client.post("/v2/namespaces/docs", json={"deletes": ["a"]})

    assert response.status_code == 200
    assert response.json()["deleted_ids"] == ["a"]
    rows = client.post(
        "/v2/namespaces/docs/query",
        json={"rank_by": ["vector", "ANN", [1.0, 0.0]], "limit": 10},
    ).json()["rows"]
    assert "a" not in [row["id"] for row in rows]


def test_missing_rank_by_is_400():
    client.post("/v2/namespaces/docs", json={"upsert_rows": [{"id": 1, "vector": [1.0, 0.0]}]})

    assert client.post("/v2/namespaces/docs/query", json={}).status_code == 400


def test_write_rejects_a_ragged_vector():
    response = client.post(
        "/v2/namespaces/docs",
        json={
            "upsert_rows": [
                {"id": 1, "vector": [1.0, 0.0]},
                {"id": 2, "vector": [1.0, 0.0, 3.0]},
            ]
        },
    )

    assert response.status_code == 400


def test_metadata_for_unknown_namespace_is_404():
    assert client.get("/v2/namespaces/ghost/metadata").status_code == 404


def test_top_k_is_accepted_as_an_alias_for_limit():
    client.post(
        "/v2/namespaces/docs",
        json={
            "upsert_rows": [
                {"id": 1, "vector": [1.0, 0.0]},
                {"id": 2, "vector": [0.0, 1.0]},
            ],
            "distance_metric": "cosine_distance",
        },
    )

    body = client.post(
        "/v2/namespaces/docs/query",
        json={"rank_by": ["vector", "ANN", [1.0, 0.0]], "top_k": 1},
    ).json()

    assert len(body["rows"]) == 1
