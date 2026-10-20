"""Executor boundary tests: never execute generated code or contact a judge."""

from unittest.mock import Mock

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.ai import answer, grade_response
from ddsr_bench.benchmarks.critpt.evaluation.ai.runtime import GraderTimeout


@pytest.fixture(autouse=True)
def forbid_local(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Injected execution must not fall back to host execution")

    monkeypatch.setattr(answer, "_execute_answer_code", forbidden)


def problem(kind, expected):
    call = {"primitive": kind, "target": "answer", "expected": expected}
    if kind == "floating_numeric":
        call["precision"] = {"kind": "decimal_places", "digits": 3}
        call["args"] = {"abs_tol": 0.001}
    return {
        "problem_id": "private-id",
        "problem_statement": "Find the value.",
        "public_solver_output_instructions": "Return answer().",
        "answer_contract": {"answer_type": kind},
        "grading_plan": {"calls": [call]},
    }


@pytest.mark.parametrize(
    "kind,actual,expected,passed",
    [
        ("exact", "2", "2", True),
        ("exact", "3", "2", False),
        ("symbolic", "x+x", "2*x", True),
        ("categorical", {"A", "B"}, ["B", "A"], True),
        ("floating_numeric", 1.25, 1.25, True),
    ],
)
def test_executor(kind, actual, expected, passed):
    # Code is validated but not evaluated: the fake worker owns the return value.
    code = "def answer():\n    return 1.25"
    response = f"Derivation.\n```python\n{code}\n```"
    execute = Mock(return_value=actual)
    report = grade_response(
        problem(kind, expected), response, execute=execute, judge_enabled=False
    )
    execute.assert_called_once_with(code)
    assert report.passed is passed
    assert report.score == float(passed)
    assert report.grading_status == ("pass" if passed else "fail")


def test_validation_before_execution():
    execute = Mock()
    report = grade_response(
        problem("exact", "2"),
        "```python\nimport os\ndef answer():\n    return os.getcwd()\n```",
        execute=execute,
        judge_enabled=False,
    )
    execute.assert_not_called()
    assert report.grading_status == "not_graded"


@pytest.mark.parametrize("error", [GraderTimeout(), RuntimeError("worker failed")])
def test_execution_failure(error):
    execute = Mock(side_effect=error)
    report = grade_response(
        problem("exact", "2"),
        "```python\ndef answer():\n    return 2\n```",
        execute=execute,
        judge_enabled=False,
    )
    execute.assert_called_once()
    assert report.grading_status == "grader_error"
    assert report.score is None
