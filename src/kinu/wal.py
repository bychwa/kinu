"""WAL segment format.

One file per commit. S3 has no append, so "appending to the log" means
"creating a new object"; the directory listing is the log.

Layout::

    0x00  HEADER    32 B   magic, version, flags, row_count, dim, metric
    0x20  VECTORS   n*D*4  little-endian f32, FIXED STRIDE
    ...   TEXT      utf-8  newline-joined (whitespace-normalized)
    ...   IDS       utf-8  newline-joined
    ...   ATTRS     utf-8  newline-joined JSON objects
    -84   FOOTER    84 B   9 x u64 block offsets/lengths, crc32, magic

Two properties carry the design:

* **Fixed stride on the vector block.** Row ``i`` starts at
  ``vec_off + i*dim*4``. One multiply, no scan. Reading row 42,000 of a
  100k-row segment costs one 4 KB ranged read, not a parse of 99,999 rows.
* **The footer is written last and is a fixed size.** A reader issues a
  single 84-byte suffix read to get it, then asks for exactly the columns
  it needs.

Columns are separate so a query that doesn't return ``text`` doesn't pay
for the text bytes. That shows up directly in ``performance.bytes_fetched``.

``crc32`` covers everything before it, so corruption is detectable. M0 does
not verify it on every read (that would mean reading the whole file); a
``kinu fsck`` command will.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

MAGIC = b"KINUWAL1"
FOOTER_MAGIC = b"KINUFTR1"

#       magic  ver  flags  row_count  dim  metric  pad
HEADER = struct.Struct("<8sIIQIB3x")
FOOTER_OFFSETS = struct.Struct("<9Q")  # vec_off len txt_off len id_off len attr_off len rows
FOOTER_SIZE = FOOTER_OFFSETS.size + 4 + 8  # + crc32 + FOOTER_MAGIC == 84

METRIC_CODES = {"cosine_distance": 0, "euclidean_squared": 1}
METRIC_NAMES = {v: k for k, v in METRIC_CODES.items()}


class CorruptSegment(ValueError):
    pass


@dataclass(frozen=True)
class Footer:
    vec_off: int
    vec_len: int
    txt_off: int
    txt_len: int
    id_off: int
    id_len: int
    attr_off: int
    attr_len: int
    row_count: int
    crc32: int

    @property
    def dim(self) -> int:
        return self.vec_len // (self.row_count * 4) if self.row_count else 0


@dataclass
class Segment:
    footer: Footer
    ids: list[Any]
    vectors: np.ndarray
    attrs: list[dict]
    texts: list[str] | None


def encode(rows: list[dict], dim: int, metric: str = "cosine_distance") -> bytes:
    """Serialize a batch of rows into one WAL segment."""
    if not rows:
        raise ValueError("cannot encode an empty batch")
    if metric not in METRIC_CODES:
        raise ValueError(f"unsupported distance metric: {metric}")

    n = len(rows)
    vectors = np.zeros((n, dim), dtype="<f4")
    for i, row in enumerate(rows):
        vector = row.get("vector")
        if vector is None:
            raise ValueError(f"row {i} has no vector")
        if len(vector) != dim:
            raise ValueError(f"row {i}: expected {dim} dims, got {len(vector)}")
        vectors[i] = np.asarray(vector, dtype=np.float32)

    ids = b"\n".join(json.dumps(row["id"]).encode() for row in rows)
    texts = "\n".join(str(row.get("text") or "") for row in rows).encode()
    attrs = "\n".join(
        json.dumps(
            {k: v for k, v in row.items() if k not in ("id", "vector", "text")},
            separators=(",", ":"),
        )
        for row in rows
    ).encode()

    out = bytearray()
    out += HEADER.pack(MAGIC, 1, 0, n, dim, METRIC_CODES[metric])
    vec_off = len(out)
    out += vectors.tobytes()
    txt_off = len(out)
    out += texts
    id_off = len(out)
    out += ids
    attr_off = len(out)
    out += attrs

    offsets = FOOTER_OFFSETS.pack(
        vec_off, n * dim * 4, txt_off, len(texts), id_off, len(ids), attr_off, len(attrs), n
    )
    body = bytes(out) + offsets
    return body + struct.pack("<I", zlib.crc32(body)) + FOOTER_MAGIC


def read_footer(store, key: str) -> Footer:
    """Read only the footer, with one suffix ranged read."""
    raw = store.get(key, start=-FOOTER_SIZE)
    if len(raw) != FOOTER_SIZE:
        raise CorruptSegment(f"{key}: truncated (expected {FOOTER_SIZE}B footer)")
    if raw[-8:] != FOOTER_MAGIC:
        raise CorruptSegment(f"{key}: bad footer magic")
    fields = FOOTER_OFFSETS.unpack(raw[: FOOTER_OFFSETS.size])
    (crc,) = struct.unpack("<I", raw[FOOTER_OFFSETS.size : FOOTER_OFFSETS.size + 4])
    return Footer(*fields, crc32=crc)


def read_rows(store, key: str, want_text: bool = True) -> Segment:
    """Read one segment. Columns not requested are never fetched."""
    footer = read_footer(store, key)
    dim = footer.dim

    vectors = np.frombuffer(
        store.get(key, footer.vec_off, footer.vec_off + footer.vec_len), dtype="<f4"
    ).reshape(footer.row_count, dim)

    ids = _json_lines(store.get(key, footer.id_off, footer.id_off + footer.id_len))
    attrs = [
        json.loads(line) if line else {}
        for line in _lines(store.get(key, footer.attr_off, footer.attr_off + footer.attr_len))
    ]
    texts: list[str] | None = None
    if want_text:
        decoded = _lines(store.get(key, footer.txt_off, footer.txt_off + footer.txt_len))
        # A batch of empty texts serializes to zero bytes, which splits to [].
        # Backfill so row indices stay aligned with the vector block.
        texts = decoded if len(decoded) == footer.row_count else [""] * footer.row_count
    return Segment(footer=footer, ids=ids, vectors=vectors, attrs=attrs, texts=texts)


def _lines(raw: bytes) -> list[str]:
    return raw.decode("utf-8").split("\n") if raw else []


def _json_lines(raw: bytes) -> list[Any]:
    """Document IDs keep their type: 7 stays 7, "7" stays "7"."""
    out: list[Any] = []
    for line in _lines(raw):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append(line)
    return out
