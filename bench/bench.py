"""Measure what a query actually costs.

Run: ``uv run python bench/bench.py``

M0 scans everything, so bytes grow linearly with namespace size. That is the
baseline the cluster index in M2 has to beat, and this script is how the README
numbers are reproduced rather than asserted.
"""

from __future__ import annotations

import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from kinu.engine import Engine
from kinu.ingest import ingest_path
from kinu.store.local import LocalStore

DIM = 1024
SIZES = (12, 1_000, 10_000)
REPEATS = 5


def namespace_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def synth(engine: Engine, namespace: str, n: int) -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, DIM), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    words = ("storage cost policy tenant index cluster object latency tier grid "
             "budget headcount remote laptop expense security incident").split()
    rows = [
        {
            "id": f"r{i}",
            "vector": vectors[i].tolist(),
            "text": " ".join(rng.choice(words, size=40)),
        }
        for i in range(n)
    ]
    for start in range(0, n, 512):
        engine.write(namespace, rows[start : start + 512])


def bench_real_corpus() -> None:
    root = Path(tempfile.mkdtemp())
    engine = Engine(LocalStore(root))
    ingest_path(engine, "docs", Path("examples/3-docs"), dim=DIM)
    total = namespace_bytes(root)

    timings = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        result = engine.query(
            "docs",
            rank_by=["vector", "ANN", ["Embed", "how much do we spend on storage"]],
            limit=10,
            include=["text"],
        )
        timings.append((time.perf_counter() - started) * 1000)

    print("real corpus (examples/3-docs)")
    print(f"  chunks            {result['performance']['approx_namespace_size']}")
    print(f"  namespace         {total:,} B")
    print(f"  fetched per query {result['performance']['bytes_fetched']:,} B "
          f"({result['performance']['bytes_fetched'] / total:.0%})")
    print(f"  requests          {result['performance']['requests']}")
    print(f"  median latency    {statistics.median(timings):.1f} ms")
    print()
    shutil.rmtree(root)


def bench_scaling() -> None:
    print(f"{'chunks':>8}  {'namespace':>12}  {'fetched':>12}  {'median':>9}")
    print("-" * 46)
    for n in SIZES:
        root = Path(tempfile.mkdtemp())
        engine = Engine(LocalStore(root))
        synth(engine, "s", n)
        total = namespace_bytes(root)

        timings = []
        fetched = 0
        for _ in range(REPEATS):
            started = time.perf_counter()
            result = engine.query(
                "s",
                rank_by=["vector", "ANN", ["Embed", "storage cost policy"]],
                limit=10,
            )
            timings.append((time.perf_counter() - started) * 1000)
            fetched = result["performance"]["bytes_fetched"]

        print(
            f"{n:>8,}  {total / 1e6:>9.2f} MB  {fetched / 1e6:>9.2f} MB  "
            f"{statistics.median(timings):>6.1f} ms"
        )
        shutil.rmtree(root)


if __name__ == "__main__":
    bench_real_corpus()
    bench_scaling()
    print()
    print("every byte of every segment is read. M2 makes this sublinear.")
