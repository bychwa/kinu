"""S3 backend — WORK IN PROGRESS.

The design is settled; the implementation is not. Everything kinu needs from
S3 maps onto a real S3 API call, so this file is a mapping table with a class
around it rather than a research problem:

    get(key, start, end)  ->  GetObject(Bucket, Key, Range="bytes=start-end")
                              a negative start becomes "bytes=-N" (suffix range)
                              *and* `len(body)` is your byte counter
    list(prefix)          ->  ListObjectsV2(Prefix=prefix), paginated by
                              ContinuationToken
    put(key, data)        ->  PutObject
    delete(key)           ->  DeleteObject
    put_if_absent(...)    ->  PutObject(..., IfNoneMatch="*")
                              ->  PreconditionFailed / 412 means "already exists"

Three honest caveats:

1. ``put_if_absent`` needs conditional writes, which S3 added in 2024. Older
   S3-compatible stores (and some R2 configurations) do not implement
   ``If-None-Match`` on PUT. Without it there is no CAS, and the WAL sequence
   allocator in ``Engine._next_seq`` becomes racy. This is the single hardest
   part of running kinu on S3, and it is a *correctness* problem, not a
   performance one.

2. Cost per request is not free. The local backend's benchmarks in
   ``bench/`` count bytes, not dollars. On S3 a cold query costs 6 PUT-class
   or GET-class requests plus transfer, and the numbers in the README will
   not transfer unchanged.

3. Latency floor. Local reads are microseconds. S3 GET is ~30-50ms warm and
   a suffix range GET is a full round trip. The three-roundtrip budget that
   the design describes only becomes *load-bearing* here.

Nothing below is enabled by default. ``Engine`` accepts any object with the
five methods, so wiring this up is a one-line change once the TODOs are done.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass


class S3Store:  # pragma: no cover - WIP
    def __init__(self, bucket: str, prefix: str = "", client=None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client

    def _k(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def get(self, key: str, start: int | None = None, end: int | None = None) -> bytes:
        # TODO: GetObject with Range. Suffix range for negative start.
        raise NotImplementedError("S3Store.get: see module docstring")

    def put(self, key: str, data: bytes) -> None:
        # TODO: PutObject
        raise NotImplementedError("S3Store.put")

    def list(self, prefix: str) -> list[str]:
        # TODO: ListObjectsV2 + ContinuationToken
        raise NotImplementedError("S3Store.list")

    def delete(self, key: str) -> None:
        # TODO: DeleteObject
        raise NotImplementedError("S3Store.delete")

    def put_if_absent(self, key: str, data: bytes) -> bool:
        # TODO: PutObject(IfNoneMatch="*"), catch 412 -> False
        raise NotImplementedError("S3Store.put_if_absent: needs conditional writes")
