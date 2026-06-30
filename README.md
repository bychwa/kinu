# kinu

*kinu* — Swahili for mortar and pestle (KEE-noo). You grind documents down; what's left is searchable.

Vector + text search that lives on object storage. Local disk today, S3 adapter WIP.
Turbopuffer-shaped API. Stateless query nodes, three roundtrips, no server to babysit.

<img src="assets/logo.svg" alt="kinu" width="420">

```
uv run kinu demo
```

## Demo

```
  parsed    3 documents → 12 chunks
  wrote     1 WAL segment(s)          53,823 B

  query: 'what happens if i lose my laptop'
    0.869  handbook         Security            Security. All laptops must have full disk encryption…
    0.952  s3-cost-notes    Request Costs       Request Costs. Requests are billed separately…
    ↳ 1.2 ms · 57,253 B fetched of 57,285 B · 8 requests · 2 roundtrips
```

No network. No API keys. No config. Three markdown files in `examples/3-docs`.

## API

Turbopuffer's v2 shape, so a client written against it reads the same here.

```bash
uv run kinu serve

curl -X POST localhost:8000/v2/namespaces/docs/query -d '{
  "rank_by": ["vector", "ANN", ["Embed", "how much do we spend on storage"]],
  "limit": 3,
  "include_attributes": ["text", "doc", "section"]
}'
```

| method | path |
|---|---|
| `POST` | `/v2/namespaces/{ns}` — `upsert_rows`, `deletes`, `distance_metric` |
| `POST` | `/v2/namespaces/{ns}/query` — `rank_by`, `filters`, `limit`, `include_attributes` |
| `GET` | `/v2/namespaces`, `/v2/namespaces/{ns}/metadata` |
| `DELETE` | `/v2/namespaces/{ns}` |

Every query returns a `performance` block. This is the point of the project:

```json
"performance": {
  "bytes_fetched": 57253, "requests": 8, "roundtrips": 2,
  "exhaustive_search_count": 12, "server_total_ms": 1.2
}
```

`bytes_fetched` is the number the architecture exists to shrink.

## How it works

One file per commit, in a namespace prefix. S3 has no append, so "appending to
the log" means "creating a new object."

```
docs/
├── meta.json         dim, distance metric
├── stats.json        corpus IDF, so rare terms dominate
└── wal/
    └── 00001.bin     header │ FIXED-STRIDE vectors │ text │ ids │ attrs │ footer
```

The footer is written last and is a fixed 84 bytes, so a reader does one suffix
read to find every column's offset, then ranged reads for only the columns it
needs. Ask for no text and no text is read.

Vectors are fixed stride: row 42,000 starts at `vec_off + 42000 * dim * 4`. One
multiply, no scan. That is what makes `bytes_fetched` a meaningful number
instead of a byte count of "whatever we happened to load."

See [DESIGN.md](DESIGN.md) for the query plan and what M2 changes.

## Numbers

`uv run python bench/bench.py`

| chunks | namespace | fetched per query | median |
|---|---|---|---|
| 12 | 57 KB | 57 KB | 1.2 ms |
| 1,000 | 4.4 MB | 4.1 MB | 2.9 ms |
| 10,000 | 44 MB | 41 MB | 24 ms |

These are local-disk numbers, so they measure CPU, not the thing that actually
hurts in production. On S3 the same queries are still **2 roundtrips**: latency
is ~2 × RTT plus transfer, and transfer is what grows.

**M0 reads everything.** That is the baseline, stated plainly. A cluster index
(M2) turns the last two rows into ~1/8 of those bytes. A search engine that
cannot tell you what it read cannot be improved.

## Status

- [x] **M0** — WAL format, byte-range reads, brute-force query, HTTP API, demo
- [ ] **M1** — CRC verification on read, `kinu fsck`
- [ ] **M2** — cluster index (`centroids.bin`, nprobe) — cuts bytes ~8x
- [ ] **M3** — BM25 inverted index, hybrid `Sum` ranking
- [ ] **M4** — manifest CAS, compaction, S3 adapter

## What this is not

Not a semantic embedder — `embed.py` is IDF-weighted feature hashing, so it
matches words, not meanings. "PTO policy" is ambiguous when another document
contains "policy" and no document contains "PTO": it will rank the wrong chunk
and that is a fair result for a lexical index.

Not a database. No transactions, no replication, no auth, no sharding.
