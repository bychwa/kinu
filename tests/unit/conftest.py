from __future__ import annotations

import pytest

from kinu.engine import Engine
from kinu.store.local import LocalStore


@pytest.fixture
def engine(tmp_path) -> Engine:
    return Engine(LocalStore(tmp_path))


@pytest.fixture
def seeded(engine: Engine) -> Engine:
    """Three rows: two share a tenant, two share a storage vector."""
    engine.write(
        "t",
        [
            {"id": "a", "vector": [1.0, 0.0], "text": "storage cost", "tenant": "acme", "page": 1},
            {"id": "b", "vector": [0.0, 1.0], "text": "pto policy", "tenant": "acme", "page": 2},
            {
                "id": "c",
                "vector": [0.9, 0.1],
                "text": "object storage",
                "tenant": "other",
                "page": 3,
            },
        ],
        distance_metric="cosine_distance",
    )
    return engine
