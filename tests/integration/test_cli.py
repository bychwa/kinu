"""End to end through the CLI.

Most cases call ``main()`` in-process so coverage sees them; one subprocess test
proves the installed entry point works, because that is the thing a user types.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kinu import cli
from kinu.cli import main

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "examples" / "3-docs"


@pytest.fixture
def data(tmp_path) -> str:
    return str(tmp_path / "data")


def test_ingest_reports_what_it_wrote(data, capsys):
    assert main(["ingest", str(CORPUS), "--data", data]) == 0

    out = capsys.readouterr().out
    assert "3 documents" in out
    assert "12 chunks" in out


def test_query_returns_the_right_document(data, capsys):
    main(["ingest", str(CORPUS), "--data", data])
    capsys.readouterr()

    assert main(["query", "how many pto days do i get", "--data", data, "--limit", "1"]) == 0

    assert "handbook" in capsys.readouterr().out


def test_query_json_shape(data, capsys):
    main(["ingest", str(CORPUS), "--data", data])
    capsys.readouterr()

    main(["query", "how much do we spend on storage", "--data", data, "--limit", "2", "--json"])

    body = json.loads(capsys.readouterr().out)
    assert len(body["rows"]) == 2
    assert body["performance"]["bytes_fetched"] > 0
    assert body["performance"]["roundtrips"] == 2


def test_second_ingest_appends_a_segment(data, capsys):
    main(["ingest", str(CORPUS), "--data", data])
    main(["ingest", str(CORPUS), "--data", data])
    capsys.readouterr()

    segments = sorted((Path(data) / "docs" / "wal").iterdir())
    assert [p.name for p in segments] == ["00001.bin", "00002.bin"]


def test_unknown_namespace_raises(data, capsys):
    main(["ingest", str(CORPUS), "--data", data])
    capsys.readouterr()

    with pytest.raises(Exception, match="nope"):
        main(["query", "storage", "--data", data, "--namespace", "nope"])


def test_demo_runs_against_the_bundled_corpus(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DEMO_DATA", tmp_path / ".kinu-demo")

    assert main(["demo"]) == 0

    out = capsys.readouterr().out
    assert "3 documents" in out
    assert "roundtrips" in out
    assert (tmp_path / ".kinu-demo" / "docs" / "wal").is_dir()


def test_no_subcommand_prints_help(capsys):
    assert main([]) == 1
    assert "search that lives on object storage" in capsys.readouterr().out


def test_entry_point_is_installed():
    """The one thing unit tests cannot prove: `python -m kinu.cli` exists."""
    result = subprocess.run(
        [sys.executable, "-m", "kinu.cli", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0
    assert "kinu" in result.stdout
