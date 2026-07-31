# kinu

*kinu* — Swahili for mortar and pestle (KEE-noo). You grind documents down; what's left is
searchable.

Vector + text search that lives on object storage. Local disk today, S3 adapter WIP.
Turbopuffer-shaped API. Stateless query nodes, two roundtrips per query, no server
to babysit.

[![ci](https://github.com/bychwa/kinu/actions/workflows/ci.yml/badge.svg)](https://github.com/bychwa/kinu/actions/workflows/ci.yml)
[![bench](https://github.com/bychwa/kinu/actions/workflows/bench.yml/badge.svg)](https://github.com/bychwa/kinu/actions/workflows/bench.yml)
[![version](https://img.shields.io/badge/version-0.1.0-c2703f)](https://github.com/bychwa/kinu/releases)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-d9a15b)](pyproject.toml)
[![coverage](https://img.shields.io/badge/coverage-85%25-2fa39a)](https://github.com/bychwa/kinu/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-8a4b2a)](LICENSE)

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
uv run kinu ingest examples/3-docs --data .kinu-demo   # load the sample corpus
uv run kinu serve --data .kinu-demo                    # http://127.0.0.1:8000

curl -s localhost:8000/v2/namespaces/docs/query \
  -H 'content-type: application/json' \
  -d '{
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

## Performance

Produced by [`bench/bench.py`](bench/bench.py) on every push that touches `src/`,
`bench/` or `examples/` — the [bench workflow](.github/workflows/bench.yml) runs it,
writes the output to the run summary and uploads `bench.txt` as an artifact. Nobody
types these numbers by hand.

Last run: [42679424660](https://github.com/bychwa/kinu/actions/runs/42679424660),
GitHub-hosted `ubuntu-latest` (x86_64).

| chunks | namespace | fetched per query | median |
|---|---|---|---|
| 12 | 0.05 MB | 0.05 MB | 0.7 ms |
| 1,000 | 4.40 MB | 4.11 MB | 7.6 ms |
| 10,000 | 43.96 MB | 41.07 MB | 74.1 ms |

Reproduce locally with `uv run python bench/bench.py` (a laptop is faster than a
shared runner; expect these to differ).

These are local-disk reads, so they measure CPU and memcpy, not the thing that
actually hurts in production. On S3 the same queries are still **2 roundtrips**:
latency is ~2 × RTT plus transfer, and transfer is what grows.

**M0 reads everything.** That is the baseline, stated plainly. A cluster index
(M2) turns the last two rows into ~1/8 of those bytes. A search engine that
cannot tell you what it read cannot be improved.

## Development

```bash
uv sync
uv run pytest tests/unit           # 28 tests, pure functions and tmp dirs
uv run pytest tests/integration    # 26 tests, HTTP, CLI, on-disk roundtrip
uv run pytest --cov=kinu --cov-fail-under=80
uv run ruff check src tests bench
```

What CI actually enforces, on every push and pull request:

| job | what it proves |
|---|---|
| `lint` | `ruff check` and `ruff format --check`, so style never lands on `main` |
| `unit` | the same tests on Python 3.11, 3.12 and 3.13 |
| `integration` | the HTTP API, the CLI, and ingest → query against files on disk |
| `coverage` | 85% today, gate at 80%, so a new branch has to bring a test |
| `package` | `uv build`, install the wheel in a clean venv, run the demo from it |
| `bench` | the table above, regenerated, never hand-edited |
| `release` | on a `v0.*` tag only: verify the tag matches the version, build, cut a GitHub release. PyPI publishing is opt-in via the `PYPI_PUBLISH` repo variable, because trusted publishing has to be configured on pypi.org first |

## Versioning

**0.1.0.** Pre-1.0 on purpose. The on-disk format changes in M2 (cluster index),
and a 1.0 would be a promise about stability that this does not honour yet. The
release workflow asserts the version is still `0.x` and that any tag matches
`pyproject.toml`, so a 1.0 has to be a deliberate decision rather than a typo.

## Status

- [x] **M0** — WAL format, byte-range reads, brute-force query, HTTP API, demo
- [x] **M0.5** — CI/CD: lint, unit + integration matrix, coverage gate, packaging, bench, release
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
