from __future__ import annotations

from decimal import Decimal
from numbers import Real
from typing import Any

import sympy as sp

from .categorical import categorical
from .numeric import exact, floating
from .symbolic import symbolic


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
