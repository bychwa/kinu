from __future__ import annotations

import numpy as np
import pytest

from kinu.embed import embed, embed_one, idf_weights, tokenize


def test_stopwords_are_dropped():
    assert tokenize("what is the pto policy") == ["pto", "policy"]


def test_stemming_folds_plurals_and_ing_forms():
    assert tokenize("laptops") == ["laptop"]
    assert tokenize("tiering") == ["tier"]
    assert tokenize("policies") == ["policy"]
    assert tokenize("reimbursed") == ["reimburs"]


def test_vectors_are_unit_length():
    vector = embed_one("how much do we spend on storage")

    assert len(vector) == 1024
    assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-5


def test_empty_text_embeds_to_zeros_not_nan():
    vector = embed_one("the and of")

    assert not np.isnan(vector).any()
    assert sum(abs(v) for v in vector) == 0.0


def test_dimension_is_configurable():
    assert embed_one("storage", dim=64).__len__() == 64


def test_identical_text_scores_1():
    vectors = embed(["object storage notes", "object storage notes"])

    assert float(vectors[0] @ vectors[1]) == pytest.approx(1.0, abs=1e-6)


def test_idf_favours_rare_terms():
    weights = idf_weights({"common": 100, "rare": 1}, n_docs=100)

    assert weights["rare"] > weights["common"]
    assert all(weight > 0 for weight in weights.values())


def test_idf_changes_the_ranking():
    """Without IDF, a rare token competes with common ones on equal footing."""
    texts = ["alpha beta beta beta", "alpha gamma"]

    plain = embed(texts, dim=256)
    weighted = embed(texts, dim=256, idf=idf_weights({"alpha": 2, "beta": 1, "gamma": 1}, 2))

    assert plain.shape == weighted.shape
    assert not np.allclose(plain, weighted), "idf weights should change the vectors"
