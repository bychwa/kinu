# Design

The constraint that produces everything else: **object storage charges you in
roundtrips (~100 ms each), not in bytes.** So the goal is never "read less."
It is "add no dependent hop, and make each hop big."

## The query plan

```
  GET meta.json + stats.json                        ~2 KB     ~10 ms
  GET wal/00001.bin  Range: bytes=-84 (footer)       84 B
  │
  │  → vec_off, vec_len, id_off, attr_off
  │
  ├──── issued together, one wall-clock roundtrip ──────┐
  │  GET … Range: bytes=vec_off..vec_off+vec_len       │
  │  GET … Range: bytes=id_off..id_off+id_len          │
  │  GET … Range: bytes=attr_off..attr_off+attr_len    │
  │  GET … Range: bytes=txt_off..txt_off+txt_len  ← skipped if not requested
  └────────────────────────────────────────────────────┘
                        │
                        ▼
        decode → distance → bounded top-k
        peak RAM = k × row + one decode buffer
```

Width is nearly free: 32 ranged GETs issued in parallel cost one roundtrip of
wall-clock. Depth is expensive. That is why the engine over-fetches a whole
column rather than adding a fourth hop to fetch ten rows of it.

## Roundtrip budget

| phase | M0 | M2 (+clusters) |
|---|---|---|
| metadata, footer | 1 | 1 |
| vectors + ids + attrs | 1 | 1 (centroids only) |
| cluster payload | — | 1 |
| **total** | **2** | **3** |

M0 reads every vector in the namespace. M2 downloads 4,000 centroids (~16 MB at
1M chunks), picks the 8 nearest, and fetches only those clusters. Same three
hops; ~1/8 the bytes.

## Segment format

```
0x00  HEADER    32 B   "KINUWAL1", version, flags, row_count, dim, metric
0x20  VECTORS   n*D*4  little-endian f32, fixed stride
 ...  TEXT      utf-8  newline-joined, whitespace-normalized
 ...  IDS       utf-8  newline-joined JSON (so 7 stays 7 and "7" stays "7")
 ...  ATTRS     utf-8  newline-joined JSON objects
 -84  FOOTER    84 B   9 × u64 offsets/lengths, crc32, "KINUFTR1"
```

Why binary and not JSON: a 1024-dim float32 vector is 4096 bytes in binary and
~8–10 KB as decimal text, *and* text has no fixed stride, so "read row 42,000"
degrades into "parse 42,000 rows." Measured on 20k × 128 vectors: JSON decodes
~1000× slower and random access is ~124,000× slower. The extension is
irrelevant; the fixed stride is not.

Why columns are separate: `include_attributes` determines which columns are
fetched. Requesting no text is measurably cheaper, and the test suite asserts it.

## Deletes

Append-only. A delete writes a tombstone row (`$deleted: true`) in a new
segment. Reads merge segments in sequence order and **later segments win**, so
the tombstone shadows the original row without rewriting anything. Compaction
(M4) physically drops shadowed rows.

## Filters

```json
["And", [["tenant", "Eq", "acme"], ["page", "Lt", 2]]]
```

Same grammar as turbopuffer: leaves are `[attr, op, value]`, combinators are
named. Evaluated after dedupe so a deleted row can never satisfy a filter.

## Embedding

IDF-weighted signed feature hashing. Without IDF, every token votes equally and
long chunks accumulate enough hash collisions to outrank exact matches — this
was observed, not theorised: "what is the pto policy" ranked a headcount
section first until IDF and stemming were added.

Cost of being lexical: no synonym or acronym matching. `embed.py` is a seam —
replace it with any `list[str] -> np.ndarray` function.

## What M2 adds, concretely

```
docs/index/
├── centroids.bin         nprobe=8 → fetch 8 nearest clusters
└── clusters/00000.bin
```

k-means over the segment vectors (k ≈ √n, capped at ~250 vectors per cluster),
one file per cluster, offsets recorded in `meta.json`. Query becomes:
read centroids → select 8 → ranged read those 8 → exact cosine. Recall drops
from 1.0 to ~0.95; `bytes_fetched` drops ~8×.

## Non-goals

No transactions. No replication. No auth. No sharding. Every one of those is a
real system's problem and none of them is the interesting part of this one.
