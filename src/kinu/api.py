"""HTTP API, shaped like turbopuffer's v2 namespace API.

Same paths, same request bodies, same response envelope. That is deliberate:
the interesting thing about this project is the storage and retrieval
architecture, not inventing a new API dialect. A client written against
turbopuffer should read the same way here.

The one addition is ``performance.bytes_fetched`` together with
``performance.roundtrips``. Turbopuffer reports cache and timing; kinu also
reports *how much it had to read to answer you*, because that number is the
entire argument for the design.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException

from kinu.engine import Engine, NamespaceNotFound
from kinu.store.local import LocalStore

DATA_DIR = os.environ.get("KINU_DATA", "./data")

engine = Engine(LocalStore(DATA_DIR))

app = FastAPI(
    title="kinu",
    version="0.1.0",
    description="Vector + text search on object storage.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v2/namespaces")
def list_namespaces() -> dict[str, list[str]]:
    return {"namespaces": engine.namespaces()}


@app.post("/v2/namespaces/{namespace}")
def write(namespace: str, body: dict[str, Any]) -> dict[str, Any]:
    affected = 0
    response: dict[str, Any] = {}
    try:
        if body.get("upsert_rows"):
            result = engine.write(
                namespace,
                body["upsert_rows"],
                distance_metric=body.get("distance_metric", "cosine_distance"),
            )
            affected += result["rows"]
            response["upserted_ids"] = [row["id"] for row in body["upsert_rows"]]
        if body.get("deletes"):
            affected += engine.delete(namespace, body["deletes"])["rows"]
            response["deleted_ids"] = list(body["deletes"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NamespaceNotFound as exc:
        raise HTTPException(status_code=404, detail=f"namespace {namespace!r} not found") from exc

    response["rows_affected"] = affected
    response["status"] = "OK"
    return response


@app.post("/v2/namespaces/{namespace}/query")
def query(namespace: str, body: dict[str, Any]) -> dict[str, Any]:
    rank_by = body.get("rank_by")
    if rank_by is None:
        raise HTTPException(status_code=400, detail="rank_by is required")

    limit = body.get("limit", body.get("top_k", 10))
    if isinstance(limit, dict):
        limit = limit.get("total", 10)

    try:
        return engine.query(
            namespace,
            rank_by=rank_by,
            filters=body.get("filters"),
            limit=int(limit),
            include=body.get("include_attributes"),
        )
    except NamespaceNotFound as exc:
        raise HTTPException(status_code=404, detail=f"namespace {namespace!r} not found") from exc
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/namespaces/{namespace}/metadata")
def metadata(namespace: str) -> dict[str, Any]:
    meta = engine.meta(namespace)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"namespace {namespace!r} not found")
    return {
        "namespace": namespace,
        "approx_row_count": meta.rows,
        "distance_metric": meta.distance_metric,
        "vector_dim": meta.dim,
        "index": {"unindexed_bytes": 0, "unindexed_rows": 0},
    }


@app.delete("/v2/namespaces/{namespace}")
def delete_namespace(namespace: str) -> dict[str, str]:
    for key in engine.store.list(namespace):
        engine.store.delete(key)
    return {"status": "OK"}
