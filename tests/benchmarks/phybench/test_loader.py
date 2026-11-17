import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.phybench.data.loader import (
    DATA_FILE,
    DATASET,
    GRADABLE_ONLY,
    REVISION,
    SPLIT,
    load_problem,
    load_problems,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "phybench_133.json"
CORRECTED_FIXTURE = Path(__file__).parents[2] / "fixtures" / "phybench_loca_41.json"


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


def test_corrected_problem() -> None:
    value = json.loads(CORRECTED_FIXTURE.read_text(encoding="utf-8"))
    problem = load_problem(value)

    assert problem.spec.id == "41"
    assert problem.answer is not None
    assert problem.answer.value.startswith("$$R\\left")


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
    assert DATASET == "Science-Discovery/LOCA"
    assert REVISION == "0e3b86b2ee18ab6118dd57029a3f8de444db365e"
    assert DATA_FILE == "problem_set/PHYBENCH_100/PHYBENCH_100_fixed.jsonl"
    assert SPLIT == "train"
    assert GRADABLE_ONLY is True
