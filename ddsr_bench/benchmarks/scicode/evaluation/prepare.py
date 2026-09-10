from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from ddsr_bench import __version__
from ddsr_bench.benchmarks.scicode.data.schemas import SciCodeProblem, SciCodeStep
from ddsr_bench.benchmarks.utils import Resources

_TEST = """#!/bin/sh
set -eu
python -m ddsr_bench.benchmarks.scicode.evaluation.verifier
"""


def encode_instruction(problem: SciCodeProblem) -> str:
    """Serialize public fields as JSON for Harbor's required instruction.md."""
    data = asdict(problem)
    for step in data["steps"]:
        step.pop("tests")
    return json.dumps(data, indent=2, ensure_ascii=False)


def decode_instruction(instruction: str) -> SciCodeProblem:
    """Restore a public SciCode problem from a Harbor instruction."""
    data = json.loads(instruction)
    steps = tuple(SciCodeStep(**step, tests=()) for step in data.pop("steps"))
    return SciCodeProblem(**data, steps=steps)


def _config(problem: SciCodeProblem, resources: Resources) -> str:
    name = json.dumps(f"scicode/{problem.id}")
    image = json.dumps(resources.image)
    return f"""schema_version = "1.4"
artifacts = [{{ source = "/app/solution.py", destination = "solution.py" }}]

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


def compile_problem(
    problem: SciCodeProblem, output: str | Path, resources: Resources
) -> Path:
    """Compile one complete SciCode problem into one Harbor task."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", problem.id) is None:
        raise ValueError(f"problem ID is not safe as a task directory: {problem.id!r}")
    if (
        not resources.image
        or min(resources.cpus, resources.memory_mb, resources.timeout_sec) <= 0
    ):
        raise ValueError("Harbor image and resource limits must be positive")

    task = Path(output) / problem.id
    task.mkdir(parents=True)
    (task / "environment").mkdir()
    tests = task / "tests"
    tests.mkdir()
    (task / "instruction.md").write_text(encode_instruction(problem), encoding="utf-8")
    (task / "task.toml").write_text(_config(problem, resources), encoding="utf-8")
    cases = [{"id": step.id, "tests": step.tests} for step in problem.steps]
    (tests / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    script = tests / "test.sh"
    script.write_text(_TEST, encoding="utf-8")
    script.chmod(0o755)
    return task


def compile_problems(
    problems: list[SciCodeProblem], output: str | Path, resources: Resources
) -> tuple[Path, ...]:
    ids = [problem.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("SciCode problem IDs must be unique")
    return tuple(compile_problem(problem, output, resources) for problem in problems)
