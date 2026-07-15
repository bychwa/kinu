"""Documents in, chunks out.

Chunking is the part everyone hand-waves and then regrets. Two rules that
matter more than they look:

1. **Normalize whitespace before storing.** The WAL stores text newline-joined,
   so a chunk containing a newline would corrupt the column. Normalizing at
   ingest time keeps the format simple and makes chunks render cleanly.
2. **Overlap by whole sentences.** A hard character overlap cuts mid-sentence
   and produces chunks that embed badly.

PDFs are supported when ``pymupdf`` happens to be installed, and skipped
cleanly when it is not. No hard dependency, no install failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kinu.embed import embed, idf_weights, tokenize

_WHITESPACE = re.compile(r"\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
PDF_SUFFIXES = {".pdf"}


@dataclass
class IngestStats:
    documents: int
    chunks: int
    segments: int
    bytes_written: int
    skipped: list[str]


def normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def read_document(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in PDF_SUFFIXES:
        try:
            import fitz  # pymupdf, optional
        except ImportError:
            return None
        with fitz.open(path) as doc:
            return "\n\n".join(page.get_text() for page in doc)
    return None


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown on headings.

    Chunking across a heading boundary is the most common way to get chunks
    that embed into mush: half the chunk is about PTO, half about laptops.
    Splitting first, then chunking, keeps each chunk about one thing.
    """
    headings = list(_HEADING.finditer(text))
    if not headings:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    if headings[0].start() > 0:
        preamble = text[: headings[0].start()]
        if preamble.strip():
            sections.append(("", preamble))

    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        sections.append((heading.group(1).strip(), text[heading.end() : end]))

    return sections


def chunk_text(text: str, target: int = 500, overlap: int = 100) -> list[str]:
    """Split normalized text into sentence-aligned chunks."""
    text = normalize(text)
    if not text:
        return []

    sentences = [s for s in _SENTENCE.split(text) if s]
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sentence in sentences:
        if current and length + len(sentence) > target:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_len = 0
            for previous in reversed(current):
                if tail_len + len(previous) > overlap:
                    break
                tail.insert(0, previous)
                tail_len += len(previous) + 1
            current, length = tail, tail_len

        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def ingest_path(
    engine,
    namespace: str,
    path: str | Path,
    *,
    dim: int = 1024,
    chunk_size: int = 500,
    overlap: int = 100,
    batch_rows: int = 512,
    distance_metric: str = "cosine_distance",
) -> IngestStats:
    root = Path(path).expanduser().resolve()
    files = sorted(p for p in (root.rglob("*") if root.is_dir() else [root]) if p.is_file())

    rows: list[dict] = []
    skipped: list[str] = []
    documents = 0

    for file in files:
        if file.suffix.lower() not in TEXT_SUFFIXES | PDF_SUFFIXES:
            continue
        body = read_document(file)
        if body is None:
            skipped.append(f"{file.name} (needs pymupdf for PDFs)")
            continue

        documents += 1
        doc_id = file.stem
        for section_index, (title, section_text) in enumerate(split_sections(body)):
            # Carry the heading into the chunk text. A chunk reading
            # "Full-time employees accrue 25 days..." is unfindable by the
            # word "PTO" unless the heading travels with it.
            prefix = f"{title}. " if title else ""
            for chunk_index, chunk in enumerate(chunk_text(section_text, chunk_size, overlap)):
                rows.append(
                    {
                        "id": f"{doc_id}#{section_index}-{chunk_index}",
                        "text": prefix + chunk,
                        "doc": doc_id,
                        "section": title,
                        "page": section_index + 1,
                        "chars": len(chunk),
                    }
                )

    if not rows:
        return IngestStats(documents, 0, 0, 0, skipped)

    # Two passes over the batch: first collect document frequencies, then
    # embed with them. Computing IDF from what was *already* in the namespace
    # gives an empty table on the very first ingest, and unweighted vectors
    # score on hash-collision noise.
    n_prev, existing_freq = engine.stats(namespace)
    token_sets = [set(tokenize(row["text"])) for row in rows]
    merged_freq = dict(existing_freq)
    for tokens in token_sets:
        for token in tokens:
            merged_freq[token] = merged_freq.get(token, 0) + 1
    n_total = n_prev + len(token_sets)

    vectors = embed(
        [row["text"] for row in rows],
        dim=dim,
        idf=idf_weights(merged_freq, n_total),
    )
    for row, vector in zip(rows, vectors, strict=True):
        row["vector"] = np.asarray(vector, dtype=np.float32).tolist()

    segments = 0
    written = 0
    for start in range(0, len(rows), batch_rows):
        result = engine.write(
            namespace,
            rows[start : start + batch_rows],
            distance_metric=distance_metric,
        )
        segments += 1
        written += result["bytes_written"]

    engine.put_stats(namespace, n_total, merged_freq)

    return IngestStats(documents, len(rows), segments, written, skipped)
