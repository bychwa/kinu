"""The storage seam.

Everything the engine needs from a backend is these five calls. If a backend
can do a ranged read and an exclusive create, kinu runs on it. That is the
whole portability story, and it is deliberately small.

`get` supports HTTP-style suffix ranges: a negative `start` means "the last
|start| bytes". That single convention is what lets the engine read a file's
footer without knowing the file's size — which is exactly what S3's
``Range: bytes=-84`` does.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Store(Protocol):
    """A byte-oriented object store with ranged reads."""

    def get(self, key: str, start: int | None = None, end: int | None = None) -> bytes:
        """Read ``key``, optionally a half-open byte range ``[start, end)``.

        A negative ``start`` reads from the end of the object.
        Raises ``FileNotFoundError`` when the key is absent.
        """

    def put(self, key: str, data: bytes) -> None:
        """Write ``key``, overwriting any existing value."""

    def list(self, prefix: str) -> list[str]:
        """Return every key under ``prefix``, sorted."""

    def delete(self, key: str) -> None:
        """Remove ``key`` if present. Missing keys are not an error."""

    def put_if_absent(self, key: str, data: bytes) -> bool:
        """Create ``key`` only if it does not exist.

        Returns ``True`` when written, ``False`` when the key already existed.
        This is the compare-and-swap primitive used for WAL sequencing and,
        later, for the namespace manifest.
        """


class MeteredStore:
    """Wraps a ``Store`` and counts work, so the API can report real costs.

    The `performance` block in every query response is populated from this.
    A search engine that cannot tell you how many bytes it read is a search
    engine you cannot reason about.
    """

    def __init__(self, inner: Store) -> None:
        self.inner = inner
        self.bytes_fetched = 0
        self.bytes_written = 0
        self.requests = 0

    def get(self, key: str, start: int | None = None, end: int | None = None) -> bytes:
        self.requests += 1
        data = self.inner.get(key, start, end)
        self.bytes_fetched += len(data)
        return data

    def put(self, key: str, data: bytes) -> None:
        self.requests += 1
        self.bytes_written += len(data)
        self.inner.put(key, data)

    def list(self, prefix: str) -> list[str]:
        self.requests += 1
        return self.inner.list(prefix)

    def delete(self, key: str) -> None:
        self.requests += 1
        self.inner.delete(key)

    def put_if_absent(self, key: str, data: bytes) -> bool:
        self.requests += 1
        self.bytes_written += len(data)
        return self.inner.put_if_absent(key, data)
