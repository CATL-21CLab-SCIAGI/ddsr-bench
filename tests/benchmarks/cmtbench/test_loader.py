import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.cmtbench.data.loader import (
    DATASET,
    REVISION,
    VERSION,
    load_problem,
    load_problems,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "cmtbench_39.json"


def record(index: int = 1) -> dict:
    return {
        "index": index,
        "prompt": "Find the requested expression.",
        "solution": r"$\boxed{SECRET_SOLUTION}$",
        "parameters": "",
        "functions": "",
        "type": "HF",
    }


def test_public_projection() -> None:
    problem = load_problem(record())

    assert problem.spec.id == "1"
    assert problem.spec.type == "HF"
    assert problem.answer is not None
    assert problem.answer.solution == r"$\boxed{SECRET_SOLUTION}$"
    assert "SECRET" not in str(asdict(problem.spec))


def test_real_v2_problem() -> None:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problem = load_problem(value)

    assert problem.spec.id == "39"
    assert problem.spec.type == "Other"
    assert problem.answer is not None
    assert problem.answer.parameters == "$t_1;t_2;n$"
    assert problem.answer.functions == ""
    assert problem.answer.solution.startswith(r"$\boxed{2\frac{t_2^2-t_1^2}")


def test_problem_type() -> None:
    value = record()
    value["type"] = "Unknown"

    with pytest.raises(ValueError, match="unsupported CMT-Benchmark problem type"):
        load_problem(value)


def test_unique_ids() -> None:
    with pytest.raises(ValueError, match="IDs must be unique"):
        load_problems([record(), record()])


def test_pinned_source() -> None:
    assert DATASET == "CMTBenchmark/cmt_benchmark"
    assert REVISION == "c6aa803a6ca553091095ecf07e747b8720b1fbc2"
    assert VERSION == "2.0.0"
