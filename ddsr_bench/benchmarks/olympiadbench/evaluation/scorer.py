from __future__ import annotations

import math
import re
from collections.abc import Sequence

import sympy as sp
from latex2sympy2_extended import latex2sympy

_REPLACEMENTS = {
    r"\left": "",
    r"\right": "",
    "∶": ":",
    "，": ",",
    "$": "",
    r"\approx": "=",
    r"\simeq": "=",
    r"\sim": "=",
    r"^\prime": "'",
    r"^{\prime}": "'",
    r"^\circ": "",
    "%": "",
}
_STRIP_CHARACTERS = "\n$,.:;^_=+`!@#$%^&*~，。"


def _boxed(value: str) -> str:
    results = []
    for match in re.finditer(r"\\boxed{", value):
        position, depth = match.end(), 1
        for end in range(position, len(value)):
            depth += value[end] == "{"
            depth -= value[end] == "}"
            if depth == 0:
                results.append(value[position:end])
                break
        else:
            raise ValueError("mismatched braces in boxed answer")
    if results:
        return ",".join(results) + ","
    answers = re.findall(r"\$(.*?)\$", value.strip().split("\n")[-1])
    return ",".join(answers) + "," if answers else value


def extract_answer(value: str) -> str:
    """Apply the official final-answer and LaTeX normalization rules."""
    value = _boxed(value)
    if r"\in " in value:
        value = value.split(r"\in ", 1)[1]
    for old, new in _REPLACEMENTS.items():
        value = value.replace(old, new)
    value = value.strip(_STRIP_CHARACTERS)
    return re.sub(r"\\(?:mathrm|mathbf)\{~?([^}]*)\}", r"\1", value)


def _split(value: str) -> list[str]:
    depth = start = 0
    parts = []
    for index, char in enumerate(value):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    if start < len(value):
        parts.append(value[start:].strip())
    expanded = []
    for part in parts:
        if r"\pm" in part:
            expanded.extend((part.replace(r"\pm", "+"), part.replace(r"\pm", "-")))
        else:
            expanded.append(part)
    return expanded


def _parse(value: str) -> sp.Expr:
    return sp.sympify(latex2sympy(value))


def _numeric(reference: str, prediction: str, tolerance: float) -> bool:
    expected, actual = float(reference), float(prediction)
    return any(
        abs(candidate - actual) <= tolerance * 1.01
        for candidate in (expected / 100, expected, expected * 100)
    )


def _expression(reference: str, prediction: str, tolerance: float) -> bool:
    reference = reference.split("=", 1)[-1].strip()
    prediction = prediction.split("=", 1)[-1].strip()
    expected, actual = _parse(reference), _parse(prediction)
    if expected == actual:
        return True
    expected = expected.subs(_parse(r"\pi"), math.pi)
    actual = actual.subs(_parse(r"\pi"), math.pi)
    expected_symbols, actual_symbols = expected.free_symbols, actual.free_symbols
    if bool(expected_symbols) != bool(actual_symbols):
        return False
    if not expected_symbols:
        for expression in (expected, actual):
            if isinstance(expression, sp.Pow):
                base, exponent = expression.as_base_exp()
                if (
                    not (base.is_number and exponent.is_number)
                    or abs(exponent.evalf()) > 1000
                ):
                    return False
        return (
            abs(complex(expected.evalf()) - complex(actual.evalf())) <= tolerance * 1.01
        )
    try:
        return abs(complex(sp.simplify(expected - actual).evalf())) < 1e-3
    except (TypeError, ValueError):
        return False


def _equation(reference: str, prediction: str) -> bool:
    def residual(value: str) -> sp.Expr:
        left, right = value.split("=")
        return sp.simplify(_parse(left) - _parse(right))

    expected, actual = residual(reference), residual(prediction)
    for ratio in (sp.simplify(expected / actual), sp.simplify(actual / expected)):
        if ratio.is_Integer and ratio != 0:
            return True
    return False


def _interval(reference: str, prediction: str, tolerance: float) -> bool:
    expected = reference.split(r"\cup")
    actual = prediction.split(r"\cup")
    if len(expected) != len(actual):
        return False
    for first, second in zip(expected, actual, strict=True):
        if first[0] != second[0] or first[-1] != second[-1]:
            return False
        first_items = first.strip("[]()").split(",")
        second_items = second.strip("[]()").split(",")
        if len(first_items) != len(second_items) or any(
            not _expression(left, right, tolerance)
            for left, right in zip(first_items, second_items, strict=True)
        ):
            return False
    return True


def _equal(reference: str, prediction: str, tolerance: float) -> bool:
    if reference == prediction and reference:
        return True
    if (
        reference.startswith(("(", "["))
        and reference.endswith((")", "]"))
        and prediction.startswith(("(", "["))
        and prediction.endswith((")", "]"))
    ):
        try:
            if _interval(reference, prediction, tolerance):
                return True
        except (IndexError, TypeError, ValueError):
            return False
    try:
        if _numeric(reference, prediction, tolerance):
            return True
    except ValueError:
        pass
    try:
        if not ("=" in reference and "=" in prediction) and _expression(
            reference, prediction, tolerance
        ):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return _equation(reference, prediction)
    except (TypeError, ValueError):
        return False


def equivalent(
    reference: str,
    prediction: str,
    precision: float | Sequence[float] = 1e-8,
) -> bool:
    """Match the pinned official AutoScoringJudge, reference-first."""
    try:
        reference = extract_answer(reference)
        prediction = extract_answer(prediction)
    except ValueError:
        return False
    if reference == prediction:
        return True
    expected = _split(re.sub(r"[\u4e00-\u9fff]+", "", reference))
    actual = _split(re.sub(r"[\u4e00-\u9fff]+", "", prediction))
    tolerances = list(precision) if isinstance(precision, Sequence) else [precision]
    if len(tolerances) <= 1:
        tolerances *= len(expected)
    if len(expected) != len(actual) or len(tolerances) != len(expected):
        return False
    while expected:
        reference_item = expected.pop(0)
        tolerance = float(tolerances.pop(0))
        for index, prediction_item in enumerate(actual):
            if _equal(reference_item, prediction_item, tolerance):
                actual.pop(index)
                break
        else:
            return False
    return True
