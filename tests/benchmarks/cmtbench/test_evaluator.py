import json
from pathlib import Path

from ddsr_bench.benchmarks.cmtbench.data.loader import load_problem
from ddsr_bench.benchmarks.cmtbench.data.schemas import AnswerSpec
from ddsr_bench.benchmarks.cmtbench.evaluation import evaluate

FIXTURE = Path(__file__).parents[2] / "fixtures" / "cmtbench_39.json"


def test_official_routes() -> None:
    cases = [
        (AnswerSpec(r"\boxed{(1, 2)}", "", ""), "numeric"),
        (AnswerSpec(r"\boxed{\pi}", "", ""), "parameterized"),
        (AnswerSpec(r"\boxed{x^2}", "$x$", ""), "parameterized"),
        (AnswerSpec(r"\boxed{a}", "$(a, NC)$", ""), "functional"),
    ]

    for answer, method in cases:
        result = evaluate(answer.solution, answer)
        assert (result.method, result.success, result.reward) == (method, True, 1)


def test_valid_difference() -> None:
    answer = AnswerSpec(r"\boxed{2}", "", "")
    result = evaluate(r"\boxed{3}", answer)

    assert result.success
    assert result.equivalent is False
    assert result.reward == 0


def test_parse_failure() -> None:
    result = evaluate("no boxed answer", AnswerSpec(r"\boxed{2}", "", ""))

    assert not result.success
    assert result.equivalent is None
    assert result.error


def test_real_parameterized_problem() -> None:
    problem = load_problem(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert problem.answer is not None

    result = evaluate(problem.answer.solution, problem.answer)

    assert result.method == "parameterized"
    assert result.equivalent is True
