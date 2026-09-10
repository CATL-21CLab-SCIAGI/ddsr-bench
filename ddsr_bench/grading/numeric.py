from __future__ import annotations

import math
from decimal import Decimal
from numbers import Real
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real | Decimal):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tolerance(value: float) -> float:
    number = _number(value)
    if number is None:
        raise TypeError("tolerance must be a finite real number")
    if number < 0:
        raise ValueError("tolerance cannot be negative")
    return number


def exact(actual: Any, expected: Any) -> bool:
    """Compare two finite scalar numbers without tolerance."""
    if _number(actual) is None or _number(expected) is None:
        return False
    return bool(actual == expected)


def floating(
    actual: Any,
    expected: Any,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    """Compare two finite scalar numbers using explicit tolerances."""
    relative = _tolerance(rel_tol)
    absolute = _tolerance(abs_tol)
    if relative == 0 and absolute == 0:
        return exact(actual, expected)
    left = _number(actual)
    right = _number(expected)
    return (
        left is not None
        and right is not None
        and math.isclose(left, right, rel_tol=relative, abs_tol=absolute)
    )
