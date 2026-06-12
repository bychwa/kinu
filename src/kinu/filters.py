"""Filter evaluation.

The syntax mirrors turbopuffer's: leaves are ``[attribute, operator, value]``
and combinators are ``["And", [leaf, leaf, ...]]``. Same shape, so a client
written against turbopuffer reads the same way here.

The ambiguity worth noting: ``["id", "Eq", 1]`` and ``["And", [...]]`` both
start with a string. Combinators are checked first by name, everything else
is a leaf. That is exactly how turbopuffer's grammar disambiguates too.
"""

from __future__ import annotations

import operator
from typing import Any

COMPARATORS = {
    "Eq": operator.eq,
    "NotEq": operator.ne,
    "Gt": operator.gt,
    "Gte": operator.ge,
    "Lt": operator.lt,
    "Lte": operator.le,
}

COMBINATORS = ("And", "Or", "Not")


def matches(row: dict[str, Any], flt: Any) -> bool:
    """Evaluate one filter expression against one row."""
    if not isinstance(flt, (list, tuple)) or len(flt) < 2:
        raise ValueError(f"malformed filter: {flt!r}")

    head = flt[0]

    if head == "And":
        return all(matches(row, child) for child in flt[1])
    if head == "Or":
        return any(matches(row, child) for child in flt[1])
    if head == "Not":
        return not matches(row, flt[1])

    if len(flt) < 3:
        raise ValueError(f"malformed leaf filter: {flt!r}")

    attribute, op, value = flt[0], flt[1], flt[2]
    actual = row.get(attribute)

    if op in COMPARATORS:
        if actual is None:
            return False
        try:
            return bool(COMPARATORS[op](actual, value))
        except TypeError:
            return False

    if op == "In":
        return actual in value
    if op == "NotIn":
        return actual not in value
    if op == "Contains":
        return actual is not None and value in actual
    if op == "ContainsAny":
        return actual is not None and any(item in actual for item in value)

    raise ValueError(f"unsupported filter operator: {op!r}")
