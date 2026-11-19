import json
import tomllib
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.physics.data.loader import load_problem
from ddsr_bench.benchmarks.physics.data.schemas import Problem
from ddsr_bench.benchmarks.physics.evaluation.prepare import (
    compile_problem,
    decode_instruction,
    decode_reference,
    prepare_tasks,
)
from ddsr_bench.benchmarks.utils import Resources, benchmark_config

FIXTURE = Path(__file__).parents[2] / "fixtures" / "physics_atomic_1_1.json"


def _problem() -> Problem:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return load_problem(record, "test")


def test_compiles_public_and_private_projections(tmp_path: Path) -> None:
    problem = _problem()

    task = compile_problem(problem, tmp_path)

    assert task.name.startswith("test-atomic-1-1-")
    spec = decode_instruction((task / "instruction.md").read_text(encoding="utf-8"))
    answer = decode_reference(
        (task / "tests" / "reference.json").read_text(encoding="utf-8")
    )
    config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    assert spec == problem.spec
    assert answer == problem.answer
    assert config["task"]["name"] == f"physics/{task.name}"
    assert "final_answers" not in (task / "instruction.md").read_text(encoding="utf-8")


def test_rejects_malformed_instruction() -> None:
    with pytest.raises(TypeError, match="must be an object"):
        decode_instruction("[]")


def test_rejects_unpinned_source_counts(tmp_path: Path) -> None:
    config = benchmark_config("physics") | {"expected_figures": {}}
    resources = Resources("unused", 1, 1, 1)

    with pytest.raises(ValueError, match="pinned configuration"):
        prepare_tasks(config, None, tmp_path, resources)
