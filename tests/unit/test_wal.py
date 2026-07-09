from __future__ import annotations

import numpy as np
import pytest

from kinu.store.local import LocalStore
from kinu.wal import FOOTER_SIZE, encode, read_footer, read_rows


def rows(n: int = 3, dim: int = 4) -> list[dict]:
    return [
        {"id": f"row-{i}", "vector": [float(i)] * dim, "text": f"document {i}", "page": i}
        for i in range(n)
    ]


def test_roundtrip_preserves_everything(tmp_path):
    store = LocalStore(tmp_path)
    store.put("ns/wal/00001.bin", encode(rows(), dim=4))

    segment = read_rows(store, "ns/wal/00001.bin")

    assert segment.ids == ["row-0", "row-1", "row-2"]
    assert segment.texts == ["document 0", "document 1", "document 2"]
    assert [a["page"] for a in segment.attrs] == [0, 1, 2]
    np.testing.assert_allclose(segment.vectors, [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])


def test_footer_is_a_suffix_range_read(tmp_path):
    store = LocalStore(tmp_path)
    blob = encode(rows(), dim=4)
    store.put("ns/wal/00001.bin", blob)

    footer = read_footer(store, "ns/wal/00001.bin")

    assert footer.row_count == 3
    assert footer.dim == 4
    assert footer.vec_len == 3 * 4 * 4
    assert len(blob) > FOOTER_SIZE


def test_vector_block_is_fixed_stride(tmp_path):
    store = LocalStore(tmp_path)
    store.put("ns/wal/00001.bin", encode(rows(4, 4), dim=4))
    footer = read_footer(store, "ns/wal/00001.bin")

    stride = footer.dim * 4
    for i in range(4):
        raw = store.get(
            "ns/wal/00001.bin", footer.vec_off + i * stride, footer.vec_off + (i + 1) * stride
        )
        np.testing.assert_allclose(np.frombuffer(raw, dtype="<f4"), [float(i)] * 4)


def test_reading_without_text_never_fetches_the_text_column(tmp_path):
    from kinu.store.base import MeteredStore

    store = MeteredStore(LocalStore(tmp_path))
    store.put("ns/wal/00001.bin", encode(rows(2, 4), dim=4))

    footer = read_footer(store, "ns/wal/00001.bin")
    assert footer.txt_len > 0

    store.bytes_fetched = 0
    segment = read_rows(store, "ns/wal/00001.bin", want_text=False)

    assert segment.texts is None
    expected = FOOTER_SIZE + footer.vec_len + footer.id_len + footer.attr_len
    assert store.bytes_fetched == expected


def test_rejects_empty_batch():
    with pytest.raises(ValueError):
        encode([], dim=4)


def test_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        encode([{"id": "a", "vector": [1.0, 2.0]}], dim=4)
