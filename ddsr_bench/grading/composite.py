from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real


def _checks(checks: Sequence[bool]) -> None:
    if not checks:
        raise ValueError("at least one grading check is required")
    if any(type(check) is not bool for check in checks):
        raise TypeError("grading checks must be booleans")


def all_required(checks: Sequence[bool]) -> bool:
    """Pass only when every grading check passes."""
    _checks(checks)
    return all(checks)


def weighted(checks: Sequence[bool], weights: Sequence[Real]) -> float:
    """Return the passed fraction of finite, non-negative weights."""
    _checks(checks)
    if len(checks) != len(weights):
        raise ValueError("checks and weights must have the same length")
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, Real)
        or not math.isfinite(weight)
        or weight < 0
        for weight in weights
    ):
        raise ValueError("weights must be finite, non-negative real numbers")
    total = sum(weights)
    if total == 0:
        raise ValueError("weights must have a positive total")
    return float(sum(weight for check, weight in zip(checks, weights) if check) / total)
