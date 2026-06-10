"""A dependency-free embedder.

This is **signed feature hashing with IDF weighting**. Each token maps to a
bucket and a sign, contributes ``idf * (1 + log tf)``, and the document vector
is L2 normalized. It is not a semantic embedding — it does not know that "car"
and "automobile" are related — but cosine similarity over these vectors is a
genuine lexical similarity, so retrieval works.

Why IDF is not optional here
----------------------------
Signed hashing gives every token an equal vote, so a long chunk of unrelated
text accumulates enough hash collisions to outscore a short, exact match.
Before IDF, "what is the pto policy" ranked an annual-report section about
headcount above the actual PTO section, purely on collision noise. Weighting
each token by inverse document frequency amplifies rare terms ("pto") and
suppresses the common ones that cause the noise. This is the same reason BM25
needs IDF; it is not a tuning knob.

Drift: document vectors are embedded with the IDF snapshot available at ingest
time. Ingest more documents later and older vectors use slightly stale weights.
Real systems re-embed on drift; for a demo corpus the drift is immaterial, and
``kinu`` re-embeds whenever a namespace is rebuilt from scratch.

Why this and not an API call:

* ``uv run kinu demo`` works offline, with no keys, in a fresh clone.
* Tests are deterministic, so ranking assertions are stable.
* It keeps the project about *retrieval architecture*, not about which
  embedding vendor you signed up for.

Swap it for anything that returns ``list[list[float]]`` — an OpenAI/Voyage
call, a local sentence-transformers model, whatever. The engine only cares
that vectors are dense, uniform-length, and comparable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from hashlib import blake2b

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "we",
        "our",
        "you",
        "your",
        "do",
        "does",
        "did",
        "not",
        "no",
        "but",
        "if",
    )
)


def _stem(token: str) -> str:
    """Crude suffix folding, enough to make plurals and -ing forms collide.

    Without this, "laptop" does not match "laptops" and "tier" does not match
    "tiering". A real stemmer (Porter/Snowball) is better; this is four rules
    that cover most of English retrieval failures at a fraction of the cost.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(token) for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


def idf_weights(document_freq: dict[str, int], n_docs: int) -> dict[str, float]:
    """Smoothed inverse document frequency, always positive."""
    return {token: math.log((n_docs + 1) / (df + 1)) + 1.0 for token, df in document_freq.items()}


def _bucket(token: str, dim: int) -> tuple[int, float]:
    digest = blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little") % dim, (1.0 if digest[4] & 1 else -1.0)


def embed(texts: list[str], dim: int = 1024, idf: dict[str, float] | None = None) -> np.ndarray:
    """Embed texts into ``dim``-dimensional unit vectors."""
    out = np.zeros((len(texts), dim), dtype=np.float32)

    for row, text in enumerate(texts):
        for token, count in Counter(tokenize(text)).items():
            bucket, sign = _bucket(token, dim)
            weight = idf.get(token, 1.0) if idf else 1.0
            # Sublinear term frequency: the 10th occurrence matters less than
            # the 2nd. Same saturation idea as BM25's k1 parameter.
            out[row, bucket] += sign * weight * (1.0 + math.log(count))

        norm = float(np.linalg.norm(out[row]))
        if norm:
            out[row] /= norm

    return out


def embed_one(text: str, dim: int = 1024, idf: dict[str, float] | None = None) -> list[float]:
    return embed([text], dim=dim, idf=idf)[0].tolist()
