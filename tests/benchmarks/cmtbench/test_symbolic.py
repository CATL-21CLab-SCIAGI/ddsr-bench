import json
from pathlib import Path

import numpy as np
import pytest

from ddsr_bench.benchmarks.cmtbench.data.loader import load_problem
from ddsr_bench.benchmarks.cmtbench.evaluation.scorer import (
    evaluate_parameterized,
    parse_symbolic,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "cmtbench_39.json"


def test_real_parameterized_solution() -> None:
    problem = load_problem(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert problem.answer is not None

    parsed = parse_symbolic(problem.answer.solution, problem.answer.parameters)
    values = evaluate_parameterized(problem.answer.solution, problem.answer.parameters)

    assert parsed.normalized == ("2(t_2^(2)-t_1^(2))/(t_2)(t_1/t_2)^n",)
    assert str(parsed.expressions[0]) == ("(t_1/t_2)**n*(-2*t_1**2 + 2*t_2**2)/t_2")
    assert values == pytest.approx((1.071245826023085,))


def test_uses_last_equality() -> None:
    parsed = parse_symbolic(r"\boxed{x=y=2x}", "$x;y$")

    assert str(parsed.expressions[0]) == "2*x"


def test_lexes_declared_symbols() -> None:
    parsed = parse_symbolic(r"\boxed{tq;i}", "$t;q;i$")

    assert parsed.normalized == ("t*q", "I")
    assert tuple(map(str, parsed.expressions)) == ("q*t", "I")


def test_preserves_upstream_flattening_behavior() -> None:
    parsed = parse_symbolic(r"\boxed{p_i q_j}", "$(p_i, NC);(q_j, NC)$")

    assert parsed.parameters["p_i"].is_commutative is False
    assert all(symbol.is_commutative for symbol in parsed.expressions[0].free_symbols)


def test_preserves_global_random_state() -> None:
    np.random.seed(7)
    first = np.random.random()
    evaluate_parameterized(r"\boxed{x^2}", "$x$")
    second = np.random.random()

    np.random.seed(7)
    assert (first, second) == (np.random.random(), np.random.random())
