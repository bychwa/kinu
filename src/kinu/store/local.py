"""Local filesystem backend.

Keys are relative paths. A ranged read is ``seek`` + ``read``; an exclusive
create is ``O_CREAT | O_EXCL``. Both map 1:1 onto their object-store
equivalents, which is why this backend is a real adapter and not a mock.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path


class LocalStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- internals ---------------------------------------------------------
    def _path(self, key: str) -> Path:
        candidate = (self.root / key.lstrip("/")).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"key escapes store root: {key!r}")
        return candidate

    def _key(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # -- Store -------------------------------------------------------------
    def get(self, key: str, start: int | None = None, end: int | None = None) -> bytes:
        path = self._path(key)
        if start is None and end is None:
            return path.read_bytes()

        size = path.stat().st_size
        lo = 0 if start is None else (max(0, size + start) if start < 0 else start)
        hi = size if end is None else max(lo, size + end if end < 0 else end)
        with path.open("rb") as handle:
            handle.seek(lo)
            return handle.read(max(0, hi - lo))

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return []
        if base.is_file():
            return [self._key(base)]
        return sorted(
            self._key(p) for p in base.rglob("*") if p.is_file() and not p.name.endswith(".tmp")
        )

    def delete(self, key: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._path(key).unlink()

    def put_if_absent(self, key: str, data: bytes) -> bool:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return True
