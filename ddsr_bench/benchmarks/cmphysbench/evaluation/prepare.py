from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ddsr_bench import __version__
from ddsr_bench.benchmarks.cmphysbench.data.loader import (
    DATASET,
    REVISION,
    load_split,
)
from ddsr_bench.benchmarks.cmphysbench.data.schemas import Problem, ProblemSpec
from ddsr_bench.benchmarks.utils import Resources

_TEST = """#!/bin/sh
set -eu
python -m ddsr_bench.benchmarks.cmphysbench.evaluation.verifier
"""


def encode_instruction(problem: ProblemSpec) -> str:
    """Serialize the public model input for a Harbor task."""
    return json.dumps(asdict(problem), indent=2, ensure_ascii=False)


def decode_instruction(instruction: str) -> ProblemSpec:
    """Restore public model input from a Harbor task."""
    return ProblemSpec(**json.loads(instruction))


def _config(problem: Problem, resources: Resources) -> str:
    name = json.dumps(f"cmphysbench/{problem.spec.id}")
    image = json.dumps(resources.image)
    return f"""schema_version = "1.4"
artifacts = [{{ source = "/app/answer.txt", destination = "answer.txt" }}]

[task]
name = {name}
version = "{__version__}"

[agent]
timeout_sec = {resources.timeout_sec}

[verifier]
timeout_sec = {resources.timeout_sec}
network_mode = "no-network"

[environment]
docker_image = {image}
workdir = "/app"
network_mode = "no-network"
cpus = {resources.cpus}
memory_mb = {resources.memory_mb}
"""


def compile_problem(problem: Problem, output: str | Path, resources: Resources) -> Path:
    """Normalize one gradeable CMPhysBench problem as a Harbor task."""
    if problem.answer is None:
        raise ValueError(f"CMPhysBench problem {problem.spec.id!r} has no reference")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", problem.spec.id) is None:
        raise ValueError(
            f"problem ID is not safe as a task directory: {problem.spec.id!r}"
        )
    if (
        not resources.image
        or min(resources.cpus, resources.memory_mb, resources.timeout_sec) <= 0
    ):
        raise ValueError("Harbor image and resource limits must be positive")

    task = Path(output) / problem.spec.id
    task.mkdir(parents=True)
    (task / "environment").mkdir()
    tests = task / "tests"
    tests.mkdir()
    (task / "instruction.md").write_text(
        encode_instruction(problem.spec), encoding="utf-8"
    )
    (task / "task.toml").write_text(_config(problem, resources), encoding="utf-8")
    reference = {
        "answer": problem.answer.value,
        "answer_type": problem.spec.answer_type,
    }
    (tests / "reference.json").write_text(
        json.dumps(reference, ensure_ascii=False), encoding="utf-8"
    )
    script = tests / "test.sh"
    script.write_text(_TEST, encoding="utf-8")
    script.chmod(0o755)
    return task


def compile_problems(
    problems: list[Problem], output: str | Path, resources: Resources
) -> tuple[Path, ...]:
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("CMPhysBench problem IDs must be unique")
    return tuple(compile_problem(problem, output, resources) for problem in problems)


def prepare_tasks(
    config: Mapping[str, Any],
    _source: str | Path | None,
    output: str | Path,
    resources: Resources,
) -> tuple[Path, ...]:
    """Load the pinned default dataset and compile its gradeable problems."""
    if (
        config.get("name") != "cmphysbench"
        or config.get("dataset") != DATASET
        or config.get("revision") != REVISION
        or not isinstance(config.get("split"), str)
        or config.get("gradable_only") is not True
    ):
        raise ValueError("CMPhysBench preparation requires its pinned configuration")
    problems = load_split(
        split=config["split"],
        revision=config["revision"],
        gradable_only=True,
    )
    return compile_problems(problems, output, resources)
