from __future__ import annotations

from typing import Any

import sympy as sp


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
