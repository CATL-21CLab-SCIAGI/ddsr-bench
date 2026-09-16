import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.phybench.data.loader import (
    DATA_FILE,
    REVISION,
    load_problem,
    load_problems,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "phybench_133.json"


def record(problem_id: int = 10, answer: str = "x^2") -> dict:
    return {
        "id": problem_id,
        "tag": "MECHANICS",
        "content": "Find the requested symbolic expression.",
        "solution": "SECRET_SOLUTION" if answer else "",
        "answer": answer,
    }


def test_public_projection() -> None:
    problem = load_problem(record())

    assert problem.spec.id == "10"
    assert problem.spec.tag == "MECHANICS"
    assert problem.answer is not None
    assert problem.answer.value == "x^2"
    assert "SECRET" not in str(asdict(problem.spec))


def test_real_problem() -> None:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problem = load_problem(value)

    assert problem.spec.id == "133"
    assert problem.answer is not None
    assert problem.answer.value == "$$\\frac{v^3}{R v_x}$$"


def test_gradable_filter() -> None:
    records = [record(10), record(11, "")]

    assert [problem.spec.id for problem in load_problems(records)] == ["10"]
    assert len(load_problems(records, gradable_only=False)) == 2


def test_reference_pair() -> None:
    value = record()
    value["solution"] = ""

    with pytest.raises(ValueError, match="both solution and answer"):
        load_problem(value)


def test_unique_ids() -> None:
    with pytest.raises(ValueError, match="IDs must be unique"):
        load_problems([record(), record()])


def test_pinned_source() -> None:
    assert REVISION == "d6d91c787b7abb865eb2490a328bf85a9f5095f0"
    assert DATA_FILE == "PHYBench-questions_v1.json"
