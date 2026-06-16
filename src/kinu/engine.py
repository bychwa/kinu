"""The engine: write path, read path, and an honest performance report.

M0 deliberately has no cluster index. Every query scans every WAL segment
end to end. That is *incorrect as a strategy and correct as a baseline*: it
gives you a working system, real byte counts, and a number to beat. The
cluster index in M2 exists to make ``bytes_fetched`` fall by an order of
magnitude, and the README quotes both.

What M0 already gets right:

* one file per commit, fixed-stride vectors, footer-indexed columns
* columns fetched independently — asking for no text means reading no text
* byte and request accounting surfaced through the API
* tombstones, so deletes are real and compaction can clean them later
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from kinu.embed import embed_one, idf_weights
from kinu.filters import matches
from kinu.store.base import MeteredStore, Store
from kinu.wal import encode, read_rows


@dataclass
class Meta:
    dim: int
    distance_metric: str = "cosine_distance"
    rows: int = 0


class NamespaceNotFound(KeyError):
    pass


class Engine:
    def __init__(self, store: Store) -> None:
        self.store = store

    # ---- keys ------------------------------------------------------------
    @staticmethod
    def meta_key(namespace: str) -> str:
        return f"{namespace}/meta.json"

    @staticmethod
    def wal_prefix(namespace: str) -> str:
        return f"{namespace}/wal"

    @staticmethod
    def stats_key(namespace: str) -> str:
        return f"{namespace}/stats.json"

    def namespaces(self) -> list[str]:
        keys = self.store.list("")
        return sorted({k.split("/", 1)[0] for k in keys if "/" in k})

    def meta(self, namespace: str) -> Meta | None:
        try:
            raw = self.store.get(self.meta_key(namespace))
        except FileNotFoundError:
            return None
        return Meta(**json.loads(raw))

    # ---- corpus statistics ------------------------------------------------
    def stats(self, namespace: str, store=None) -> tuple[int, dict[str, int]]:
        store = store or self.store
        try:
            raw = store.get(self.stats_key(namespace))
        except FileNotFoundError:
            return 0, {}
        payload = json.loads(raw)
        return int(payload.get("n", 0)), payload.get("df", {})

    def idf(self, namespace: str, store=None) -> dict[str, float]:
        n_docs, document_freq = self.stats(namespace, store)
        return idf_weights(document_freq, n_docs)

    def add_stats(self, namespace: str, documents: list[set[str]]) -> None:
        n_docs, document_freq = self.stats(namespace)
        for tokens in documents:
            for token in tokens:
                document_freq[token] = document_freq.get(token, 0) + 1
        self.put_stats(namespace, n_docs + len(documents), document_freq)

    def put_stats(self, namespace: str, n_docs: int, document_freq: dict[str, int]) -> None:
        payload = {"n": n_docs, "df": document_freq}
        self.store.put(self.stats_key(namespace), json.dumps(payload).encode())

    # ---- write -----------------------------------------------------------
    def write(
        self,
        namespace: str,
        rows: list[dict[str, Any]],
        distance_metric: str = "cosine_distance",
    ) -> dict[str, int]:
        if not rows:
            raise ValueError("write requires at least one row")

        for i, row in enumerate(rows):
            if "id" not in row:
                raise ValueError(f"row {i} is missing 'id'")
            if "vector" not in row:
                raise ValueError(f"row {i} is missing 'vector'")

        dim = len(rows[0]["vector"])
        for i, row in enumerate(rows):
            if len(row["vector"]) != dim:
                raise ValueError(f"row {i}: expected {dim} dimensions, got {len(row['vector'])}")

        meta = self.meta(namespace)
        if meta is None:
            meta = Meta(dim=dim, distance_metric=distance_metric)
            self.store.put(self.meta_key(namespace), json.dumps(asdict(meta)).encode())
        elif meta.dim != dim:
            raise ValueError(f"namespace {namespace!r} is {meta.dim}-dimensional, got {dim}")

        blob = encode(rows, dim, meta.distance_metric)
        seq = self._next_seq(namespace)
        key = f"{self.wal_prefix(namespace)}/{seq:05d}.bin"
        if not self.store.put_if_absent(key, blob):
            raise RuntimeError(f"WAL sequence race on {key}")

        meta.rows += len(rows)
        self.store.put(self.meta_key(namespace), json.dumps(asdict(meta)).encode())
        return {"wal_seq": seq, "rows": len(rows), "bytes_written": len(blob)}

    def delete(self, namespace: str, ids: list[Any]) -> dict[str, int]:
        meta = self.meta(namespace)
        if meta is None:
            raise NamespaceNotFound(namespace)
        tombstones = [
            {"id": doc_id, "vector": [0.0] * meta.dim, "$deleted": True} for doc_id in ids
        ]
        return self.write(namespace, tombstones)

    def _next_seq(self, namespace: str) -> int:
        keys = self.store.list(self.wal_prefix(namespace))
        seqs = [
            int(key.rsplit("/", 1)[-1].split(".")[0])
            for key in keys
            if key.endswith(".bin") and key.rsplit("/", 1)[-1].split(".")[0].isdigit()
        ]
        return max(seqs) + 1 if seqs else 1

    # ---- read ------------------------------------------------------------
    def query(
        self,
        namespace: str,
        *,
        rank_by: list[Any],
        filters: Any = None,
        limit: int = 10,
        include: list[str] | bool | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        meter = MeteredStore(self.store)

        try:
            meta_raw = meter.get(self.meta_key(namespace))
        except FileNotFoundError as exc:
            raise NamespaceNotFound(namespace) from exc
        meta = Meta(**json.loads(meta_raw))

        if not isinstance(rank_by, (list, tuple)) or len(rank_by) < 3:
            raise ValueError("rank_by must be [attribute, 'ANN', vector]")
        attribute, kind, payload = rank_by[0], str(rank_by[1]).upper(), rank_by[2]
        if kind not in ("ANN", "KNN"):
            raise NotImplementedError(f"ranking function {kind!r} lands in M3")
        if attribute != "vector":
            raise ValueError("M0 supports rank_by on the 'vector' attribute only")

        # turbopuffer allows native embedding inside the query. Same shape
        # here, and it is the only way to get correct client-side IDF
        # weighting: the weights live in the namespace, not in the caller.
        if isinstance(payload, (list, tuple)) and payload and payload[0] == "Embed":
            if len(payload) < 2:
                raise ValueError("['Embed', text] requires query text")
            query_vector = np.asarray(
                embed_one(str(payload[1]), dim=meta.dim, idf=self.idf(namespace, meter)),
                dtype=np.float32,
            )
        else:
            query_vector = np.asarray(payload, dtype=np.float32)

        if query_vector.shape != (meta.dim,):
            raise ValueError(f"query vector has {query_vector.size} dims, namespace has {meta.dim}")

        want_all = include is True
        wanted = set() if include in (None, False) else set(include)
        want_text = want_all or "text" in wanted

        # LSM semantics: later segments supersede earlier ones for the same ID.
        # A tombstone in segment N removes a row written in segment N-1, which
        # is the whole reason deletes are append-only here.
        scanned = 0
        latest: dict[Any, tuple[float, dict, str | None] | None] = {}

        for key in sorted(meter.list(self.wal_prefix(namespace))):
            segment = read_rows(meter, key, want_text=want_text)
            distances = _distance(query_vector, segment.vectors, meta.distance_metric)
            scanned += segment.footer.row_count
            for i in range(segment.footer.row_count):
                doc_id = segment.ids[i]
                attributes = segment.attrs[i]
                if attributes.get("$deleted"):
                    latest[doc_id] = None
                    continue
                text = segment.texts[i] if segment.texts is not None else None
                latest[doc_id] = (float(distances[i]), attributes, text)

        candidates: list[tuple[float, Any, dict, str | None]] = []
        for doc_id, entry in latest.items():
            if entry is None:
                continue
            distance, attributes, text = entry
            if filters is not None and not matches({"id": doc_id, **attributes}, filters):
                continue
            candidates.append((distance, doc_id, attributes, text))

        candidates.sort(key=lambda candidate: candidate[0])

        rows: list[dict[str, Any]] = []
        for distance, doc_id, attributes, text in candidates[: max(0, limit)]:
            row: dict[str, Any] = {"$dist": round(distance, 6), "id": doc_id}
            for name in attributes if want_all else wanted:
                if name in attributes:
                    row[name] = attributes[name]
            if want_text and text is not None:
                row["text"] = text
            rows.append(row)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "rows": rows,
            "performance": {
                "cache_hit_ratio": 0.0,
                "server_total_ms": round(elapsed_ms, 2),
                "exhaustive_search_count": scanned,
                "approx_namespace_size": meta.rows,
                "bytes_fetched": meter.bytes_fetched,
                "requests": meter.requests,
                "roundtrips": 2,  # footer, then column blocks
                "clusters_probed": 0,
            },
            "billing": {"billable_logical_bytes_queried": meter.bytes_fetched},
        }


def _distance(query: np.ndarray, matrix: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine_distance":
        query_norm = float(np.linalg.norm(query)) or 1.0
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        return 1.0 - (matrix @ query) / (norms * query_norm)
    if metric == "euclidean_squared":
        return ((matrix - query) ** 2).sum(axis=1)
    raise ValueError(f"unsupported distance metric: {metric!r}")
