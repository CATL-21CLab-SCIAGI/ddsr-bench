from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ddsr_bench.benchmarks.prismphysics.data.schemas import AnswerSpec

from .gradephyx import simplify_latex_expr, whether_rel_latex_correct_with_units

_BLOCK = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_RELATIONS = ("=", "<", ">", r"\leq", r"\geq", r"\ll", r"\gg", r"\approx")
_FRACTION = re.compile(r"\\frac\s*\{\s*[+-]?\d+\s*\}\s*\{\s*[+-]?\d+\s*\}")
_SCIENTIFIC = (
    re.compile(
        r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*"
        r"(?:\\times|\\cdot|\*)\s*10\^(\{[-+]?\d+\}|[-+]?\d+)"
    ),
    re.compile(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))[eE]([+-]?\d+)"),
)
_DECIMAL = re.compile(r"(?<!\w)([+-]?(?:\d+\.\d+|\.\d+))")


@dataclass(frozen=True, slots=True)
class FormulaMatch:
    reference_index: int
    answer_index: int


@dataclass(frozen=True, slots=True)
class Evaluation:
    score: float
    correct: bool
    formula_count: int
    answer_formula_count: int
    direct_matches: tuple[int, ...]
    credited: tuple[int, ...]
    matches: tuple[FormulaMatch, ...]


def extract_formulas(response: str) -> tuple[str, ...]:
    """Extract and normalize relational double-dollar formula blocks."""
    formulas = (
        simplify_latex_expr(value.strip()) for value in _BLOCK.findall(response)
    )
    return tuple(
        formula
        for formula in formulas
        if any(relation in formula for relation in _RELATIONS)
    )


def _overlaps(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < right and end > left for left, right in spans)


def _exact_decimal(text: str) -> bool:
    if re.fullmatch(r"[+-]?(?:0?\.25|0?\.75)", text):
        return True
    match = re.fullmatch(r"[+-]?(?:\d+\.(\d)|\.(\d))", text)
    if match is None:
        return False
    return abs(float(text)) < 1 or (match.group(1) or match.group(2)) in ("0", "5")


def _tolerance(formula: str) -> float:
    """Match upstream's precision inferred from decimals and scientific notation."""
    spans = [match.span() for match in _FRACTION.finditer(formula)]
    tolerances = []
    for pattern in _SCIENTIFIC:
        for match in pattern.finditer(formula):
            if _overlaps(spans, *match.span()):
                continue
            mantissa = match.group(1)
            exponent = int(match.group(2).strip("{}"))
            value = float(mantissa) * 10**exponent
            decimals = len(mantissa.partition(".")[2])
            absolute = 5 * 10 ** (-decimals) * 10**exponent
            if value:
                tolerances.append(absolute / abs(value))
            spans.append(match.span())
    for match in _DECIMAL.finditer(formula):
        if _overlaps(spans, *match.span()) or _exact_decimal(match.group(1)):
            continue
        value = float(match.group(1))
        decimals = len(match.group(1).partition(".")[2])
        if value:
            tolerances.append(5 * 10 ** (-decimals) / abs(value))
    return max(tolerances, default=0.0)


def evaluate(response: str, answer: AnswerSpec, threshold: float) -> Evaluation:
    """Apply formula matching followed by PRISM ancestor-closure scoring."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    candidates = extract_formulas(response)
    direct = set()
    matches = []
    for node in answer.formulas:
        reference = node.formula.replace("$", "")
        for answer_index, candidate in enumerate(candidates):
            tolerance = max(_tolerance(reference), _tolerance(candidate), 1e-5)
            try:
                equivalent, _ = whether_rel_latex_correct_with_units(
                    reference,
                    candidate,
                    epsilon_for_equal=tolerance,
                )
            except Exception:  # noqa: BLE001, S112 - upstream skips invalid pairs
                continue
            if equivalent:
                direct.add(node.index)
                matches.append(FormulaMatch(node.index, answer_index))
                break

    dependencies = {node.index: node.dependencies for node in answer.formulas}
    credited = set()

    def credit(index: int) -> None:
        if index in credited:
            return
        credited.add(index)
        for dependency in dependencies[index]:
            credit(dependency)

    for index in direct:
        credit(index)
    score = len(credited) / len(answer.formulas) if answer.formulas else 0.0
    return Evaluation(
        score=score,
        correct=score >= threshold,
        formula_count=len(answer.formulas),
        answer_formula_count=len(candidates),
        direct_matches=tuple(sorted(direct)),
        credited=tuple(sorted(credited)),
        matches=tuple(matches),
    )
