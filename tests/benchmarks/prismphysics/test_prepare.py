import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from harbor.models.task.config import NetworkMode
from harbor.models.task.task import Task

from ddsr_bench.benchmarks.prismphysics.data.loader import load_problem
from ddsr_bench.benchmarks.prismphysics.evaluation.prepare import (
    compile_problem,
    decode_instruction,
    encode_instruction,
)
from ddsr_bench.benchmarks.utils import Resources

FIXTURE = Path(__file__).parents[2] / "fixtures" / "prismphysics_02_3006.json"


def problem():
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return load_problem(record, "02")


def test_compile(tmp_path: Path) -> None:
    source = problem()
    assert source.answer is not None
    task_path = compile_problem(
        source, tmp_path, Resources("ddsr-bench-prismphysics:test", 2, 4096, 120)
    )

    task = Task(task_path)
    assert source.spec.images
    assert decode_instruction(encode_instruction(source.spec)) == replace(
        source.spec, images=()
    )
    assert source.spec.images[0].location not in task.instruction
    assert source.answer.formulas[0].formula not in task.instruction
    reference = json.loads((task_path / "tests" / "reference.json").read_text())
    assert reference["formulas"][1]["dependencies"] == [1]
    assert reference["threshold"] == 0.8
    assert task.config.environment.network_mode == NetworkMode.NO_NETWORK
    assert task.config.environment.docker_image == "ddsr-bench-prismphysics:test"
    assert task.config.artifacts[0].source == "/app/answer.txt"
    assert (task_path / "tests" / "test.sh").stat().st_mode & 0o111


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan])
def test_rejects_invalid_threshold(tmp_path: Path, threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be between"):
        compile_problem(
            problem(),
            tmp_path,
            Resources("ddsr-bench-prismphysics:test", 2, 4096, 120),
            threshold,
        )
