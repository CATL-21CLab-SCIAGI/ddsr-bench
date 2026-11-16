from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ddsr_bench.benchmarks.cmtbench.data.schemas import AnswerSpec

from .scorer import (
    equivalent,
    equivalent_functional,
    evaluate_parameterized,
    parse_numeric,
)

Method = Literal["numeric", "parameterized", "functional"]


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Outcome of one official CMT answer comparison."""

    method: Method
    success: bool
    equivalent: bool | None
    error: str = ""

    @property
    def reward(self) -> float:
        return float(self.equivalent is True)


def _method(answer: AnswerSpec) -> Method:
    if not answer.parameters and not answer.functions:
        try:
            parse_numeric(answer.solution)
        except ValueError:
            return "parameterized"
        return "numeric"
    if answer.parameters and not answer.functions and "NC" not in answer.parameters:
        return "parameterized"
    return "functional"


def evaluate(response: str, answer: AnswerSpec) -> Evaluation:
    """Route and compare one response exactly as the official evaluation script."""
    method = _method(answer)
    try:
        if method == "numeric":
            reference = parse_numeric(answer.solution)
            model = parse_numeric(response)
            passed = equivalent(model, reference)
        elif method == "parameterized":
            reference = evaluate_parameterized(answer.solution, answer.parameters)
            model = evaluate_parameterized(response, answer.parameters)
            passed = equivalent(model, reference)
        else:
            passed = equivalent_functional(
                response,
                answer.solution,
                answer.parameters,
                answer.functions,
            )
    except Exception as error:  # noqa: BLE001 - parser failures are score data
        return Evaluation(method, False, None, str(error))
    return Evaluation(method, True, passed)
