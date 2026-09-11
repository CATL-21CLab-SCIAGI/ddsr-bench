from dataclasses import asdict

import pytest

from ddsr_bench.benchmarks.cmphysbench.data.loader import (
    REVISION,
    load_problem,
    load_problems,
)


def record(problem_id: int = 1, final_answers: list[str] | None = None) -> dict:
    return {
        "id": problem_id,
        "question": "Find the energy using $k$.",
        "answer": "SECRET_WORKED_SOLUTION",
        "final_answer": ["k^2"] if final_answers is None else final_answers,
        "answer_type": "Expression",
        "topic": "Theoretical Foundations",
        "symbol": "$k$: wave vector",
    }


def test_public_projection() -> None:
    value = record()
    value["symbol"] = ""
    problem = load_problem(value)

    assert problem.spec.id == "1"
    assert problem.spec.context == ""
    assert problem.spec.symbols == ""
    assert problem.answer is not None
    assert problem.answer.value == "k^2"
    assert "SECRET" not in str(asdict(problem.spec))


def test_gradable_filter() -> None:
    records = [record(1), record(2, [])]

    assert [problem.spec.id for problem in load_problems(records)] == ["1"]
    assert len(load_problems(records, gradable_only=False)) == 2


def test_answer_type() -> None:
    value = record()
    value["answer_type"] = "Unknown"

    with pytest.raises(ValueError, match="unsupported CMPhysBench answer type"):
        load_problem(value)


def test_unique_ids() -> None:
    with pytest.raises(ValueError, match="IDs must be unique"):
        load_problems([record(), record()])


def test_pinned_revision() -> None:
    assert REVISION == "43d185851f731e23aa5737c3667b3a9e87bf8cd1"
