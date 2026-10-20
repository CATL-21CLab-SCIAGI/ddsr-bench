"""Reference-value comparisons for the ordinary CritPt verifier."""

from __future__ import annotations

import math
from decimal import Decimal
from numbers import Real
from typing import Any

import sympy as sp


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


def categorical(actual: Any, expected: Any) -> bool:
    """Compare two non-empty multiple-choice labels."""
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    left = actual.strip().casefold()
    right = expected.strip().casefold()
    return bool(left) and left == right


def _expression(value: Any) -> sp.Expr | None:
    if isinstance(value, bool | str | bytes):
        return None
    try:
        expression = sp.sympify(value, strict=True)
    except (sp.SympifyError, TypeError, ValueError):
        return None
    if not isinstance(expression, sp.Expr):
        return None
    if expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return None
    return expression


def symbolic(actual: Any, expected: Any) -> bool:
    """Compare two finite scalar expressions algebraically."""
    left = _expression(actual)
    right = _expression(expected)
    if left is None or right is None:
        return False
    try:
        return sp.simplify(sp.together(left - right)) == 0
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def compare(
    actual: Any,
    expected: Any,
    *,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-12,
) -> bool:
    """Compare one returned value according to its reference value's type."""
    if isinstance(expected, str):
        return categorical(actual, expected)
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, float | sp.Float):
        return floating(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(expected, sp.Expr):
        return symbolic(actual, expected)
    if isinstance(expected, Real | Decimal):
        return exact(actual, expected)
    if isinstance(expected, tuple | list):
        return (
            isinstance(actual, type(expected))
            and len(actual) == len(expected)
            and bool(expected)
            and all(
                compare(left, right, rel_tol=rel_tol, abs_tol=abs_tol)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and bool(expected)
            and all(
                compare(actual[key], value, rel_tol=rel_tol, abs_tol=abs_tol)
                for key, value in expected.items()
            )
        )
    return False
