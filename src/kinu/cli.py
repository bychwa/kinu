"""Command line: ``kinu demo``, ``kinu ingest``, ``kinu query``, ``kinu serve``.

``kinu demo`` is the front door. One command, no keys, no network, no config,
deterministic output. If a project cannot be experienced in thirty seconds it
does not get experienced at all.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from kinu.engine import Engine
from kinu.ingest import ingest_path
from kinu.store.local import LocalStore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DATA = REPO_ROOT / ".kinu-demo"

DEMO_QUERIES = [
    "how much do we spend on storage",
    "how many pto days do i get",
    "what happens if i lose my laptop",
]


def _examples_dir() -> Path:
    for candidate in (REPO_ROOT / "examples" / "3-docs", Path.cwd() / "examples" / "3-docs"):
        if candidate.is_dir():
            return candidate
    raise SystemExit("examples/3-docs not found — run from the repo root")


def _rule(char: str = "─", width: int = 66) -> str:
    return char * width


# ---------------------------------------------------------------------------
def cmd_demo(args: argparse.Namespace) -> int:
    docs = _examples_dir()
    if DEMO_DATA.exists():
        shutil.rmtree(DEMO_DATA)

    engine = Engine(LocalStore(DEMO_DATA))
    print()
    print(_rule("━"))
    print("  kinu demo  ·  local disk, no network, no API keys")
    print(_rule("━"))
    print()

    started = time.perf_counter()
    stats = ingest_path(engine, "docs", docs, dim=args.dim)
    ingest_ms = (time.perf_counter() - started) * 1000

    total_bytes = sum(p.stat().st_size for p in DEMO_DATA.rglob("*") if p.is_file())

    print(f"  parsed    {stats.documents} documents → {stats.chunks} chunks")
    print(f"  wrote     {stats.segments} WAL segment(s)        {stats.bytes_written:>8,} B")
    print(f"  total     {total_bytes:>8,} B on disk            {ingest_ms:>6.0f} ms")
    if stats.skipped:
        for note in stats.skipped:
            print(f"  skipped   {note}")
    print()

    for question in DEMO_QUERIES:
        started = time.perf_counter()
        result = engine.query(
            "docs",
            rank_by=["vector", "ANN", ["Embed", question]],
            limit=3,
            include=["text", "doc", "section"],
        )
        elapsed = (time.perf_counter() - started) * 1000

        print(f"  query: {question!r}")
        for row in result["rows"]:
            snippet = " ".join(row["text"].split())[:66]
            section = (row.get("section") or "-")[:20]
            print(f"    {row['$dist']:.3f}  {row['doc']:<16} {section:<20} {snippet}")
        perf = result["performance"]
        print(
            f"    ↳ {elapsed:.1f} ms · {perf['bytes_fetched']:,} B fetched of "
            f"{total_bytes:,} B · {perf['requests']} requests · {perf['roundtrips']} roundtrips"
        )
        print()

    print(_rule("━"))
    print("  Every query above scanned every chunk. That is the baseline.")
    print("  M2 adds a cluster index; the same query then fetches ~1/8 the bytes.")
    print(_rule("━"))
    print()
    print(f"  data: {DEMO_DATA}")
    print("  next: uv run kinu serve")
    print()
    return 0


# ---------------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> int:
    engine = Engine(LocalStore(args.data))
    stats = ingest_path(
        engine,
        args.namespace,
        args.path,
        dim=args.dim,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(
        f"ingested {stats.documents} documents, {stats.chunks} chunks, "
        f"{stats.bytes_written:,} B in {stats.segments} segment(s)"
    )
    for note in stats.skipped:
        print(f"skipped: {note}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    engine = Engine(LocalStore(args.data))
    result = engine.query(
        args.namespace,
        rank_by=["vector", "ANN", ["Embed", args.text]],
        limit=args.limit,
        include=["text", "doc", "section"],
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    for row in result["rows"]:
        print(f"{row['$dist']:.3f}  {row['doc']:<16} {row['text'][:90]}")
    perf = result["performance"]
    print(f"\n{perf['bytes_fetched']:,} B fetched in {perf['server_total_ms']} ms")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    # api.py resolves its data directory at import time, and the reloader
    # re-imports it in a child process, so the variable has to be in the
    # environment rather than passed as an argument.
    os.environ["KINU_DATA"] = args.data
    print(f"kinu serving {args.data} on http://{args.host}:{args.port}")
    uvicorn.run("kinu.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kinu", description="search that lives on object storage")
    parser.add_argument("--dim", type=int, default=1024, help="embedding dimensions (default 1024)")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="ingest the bundled corpus and search it")
    demo.set_defaults(func=cmd_demo)

    ingest = sub.add_parser("ingest", help="ingest a file or directory")
    ingest.add_argument("path")
    ingest.add_argument("--namespace", default="docs")
    ingest.add_argument("--data", default="./data")
    ingest.add_argument("--chunk-size", type=int, default=500)
    ingest.add_argument("--overlap", type=int, default=100)
    ingest.set_defaults(func=cmd_ingest)

    query = sub.add_parser("query", help="run one query")
    query.add_argument("text")
    query.add_argument("--namespace", default="docs")
    query.add_argument("--data", default="./data")
    query.add_argument("--limit", type=int, default=5)
    query.add_argument("--json", action="store_true")
    query.set_defaults(func=cmd_query)

    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--data", default="./data")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    # deliver --dim to subcommands that need it, defaulting from the parent
    if not hasattr(args, "dim") or args.dim is None:
        args.dim = 1024

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
