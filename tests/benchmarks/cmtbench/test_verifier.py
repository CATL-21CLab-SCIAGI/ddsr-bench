import json
from pathlib import Path

from ddsr_bench.benchmarks.cmtbench.data.loader import load_problem
from ddsr_bench.benchmarks.cmtbench.data.schemas import AnswerSpec
from ddsr_bench.benchmarks.cmtbench.evaluation import verify

FIXTURE = Path(__file__).parents[2] / "fixtures" / "cmtbench_39.json"


def test_real_answer() -> None:
    problem = load_problem(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert problem.answer is not None

    result = verify(problem.answer.solution, problem.answer)

    assert result == {
        "reward": 1.0,
        "mode": "cmtbench",
        "status": "passed",
        "method": "parameterized",
        "success": True,
        "equivalent": True,
    }
    assert "solution" not in result


def test_difference() -> None:
    result = verify(r"\boxed{3}", AnswerSpec(r"\boxed{2}", "", ""))

    assert result["reward"] == 0
    assert result["status"] == "different"
    assert result["method"] == "numeric"


def test_parser_failure() -> None:
    result = verify("no box", AnswerSpec(r"\boxed{2}", "", ""))

    assert result["reward"] == 0
    assert result["status"] == "error"
    assert result["method"] == "numeric"
    assert result["error"]


def test_timeout() -> None:
    result = verify(
        r"\boxed{2}",
        AnswerSpec(r"\boxed{2}", "", ""),
        timeout_sec=0,
    )

    assert result["reward"] == 0
    assert result["status"] == "timeout"
    assert result["error"] == "TimeoutError"
