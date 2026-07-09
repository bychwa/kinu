from __future__ import annotations

import pytest

from kinu.filters import matches

ROW = {"tenant": "acme", "page": 3, "tags": ["a", "b"], "ratio": 0.5}


def test_eq_and_neq():
    assert matches(ROW, ["tenant", "Eq", "acme"])
    assert not matches(ROW, ["tenant", "Eq", "other"])


def test_comparisons():
    assert matches(ROW, ["page", "Gte", 3])
    assert matches(ROW, ["page", "Gt", 2])
    assert matches(ROW, ["page", "Lt", 4])
    assert matches(ROW, ["page", "Lte", 3])
    assert not matches(ROW, ["page", "Gt", 3])


def test_contains():
    assert matches(ROW, ["tags", "ContainsAny", ["b", "z"]])
    assert not matches(ROW, ["tags", "ContainsAny", ["z"]])


def test_boolean_combinators():
    assert matches(ROW, ["Or", [["page", "Lt", 1], ["tenant", "Eq", "acme"]]])
    assert matches(ROW, ["And", [["page", "Lt", 5], ["tenant", "Eq", "acme"]]])
    assert matches(ROW, ["Not", ["tenant", "Eq", "other"]])
    assert not matches(ROW, ["Not", ["tenant", "Eq", "acme"]])


def test_missing_attribute_is_false():
    assert not matches(ROW, ["nope", "Eq", "x"])


def test_unknown_operator_is_an_error():
    with pytest.raises(ValueError):
        matches(ROW, ["page", "Wat", 1])
