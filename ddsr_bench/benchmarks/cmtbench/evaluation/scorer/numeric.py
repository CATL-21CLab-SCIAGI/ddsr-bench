from __future__ import annotations

import ast

import numpy as np

from .extract import extract_solutions

Scalar = float | complex
Number = Scalar | tuple[Scalar, ...]


def parse_numeric(response: str) -> tuple[Number, ...]:
    """Parse the numeric scalar or tuple forms accepted by the official scorer."""
    parsed = []
    for extracted in extract_solutions(response):
        token = extracted.split("=")[-1].strip()
        try:
            value = ast.literal_eval(token)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"cannot parse numeric solution {token!r}") from error
        if isinstance(value, int | float):
            parsed.append(float(value))
        elif isinstance(value, tuple) and all(
            isinstance(item, int | float) for item in value
        ):
            parsed.append(tuple(float(item) for item in value))
        else:
            raise ValueError(f"numeric solution has unsupported value {value!r}")
    return tuple(parsed)


def _sort_key(value: Number) -> tuple[float, float]:
    item = value[0] if hasattr(value, "__getitem__") else value
    if isinstance(item, complex):
        return item.real, item.imag
    return float(item), 0


def equivalent(model: tuple[Number, ...], reference: tuple[Number, ...]) -> bool:
    """Compare unordered numeric results with upstream's fixed tolerances."""
    if len(model) != len(reference):
        return False
    for actual, expected in zip(
        sorted(model, key=_sort_key), sorted(reference, key=_sort_key), strict=True
    ):
        if hasattr(actual, "__len__") != hasattr(expected, "__len__"):
            return False
        if hasattr(actual, "__len__") and len(actual) != len(expected):
            return False
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-12):
            return False
    return True
