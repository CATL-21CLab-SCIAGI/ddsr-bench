from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from ddsr_bench import __version__
from ddsr_bench.benchmarks.critpt.data.schemas import Challenge, Problem, ProblemSpec
from ddsr_bench.benchmarks.utils import Resources

_TEST = """#!/bin/sh
set -eu
python -m ddsr_bench.benchmarks.critpt.evaluation.verifier
"""


def encode_instruction(problem: ProblemSpec) -> str:
    """Serialize public fields as JSON for Harbor's required instruction.md."""
    data = asdict(problem)
    data.pop("source_path")
    return json.dumps(data, indent=2, ensure_ascii=False)


def decode_instruction(instruction: str) -> ProblemSpec:
    """Restore the public problem fields from one prepared task."""
    data = json.loads(instruction)
    return ProblemSpec(**data, source_path=Path("prepared-task"))


def _config(problem: Problem, resources: Resources) -> str:
    name = json.dumps(f"critpt/{problem.spec.id}")
    image = json.dumps(resources.image)
    return f"""schema_version = "1.4"
artifacts = [{{ source = "/app/answer.py", destination = "answer.py" }}]

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
    """Compile one CritPt problem into one Harbor task directory."""
    problem_id = problem.spec.id
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", problem_id) is None:
        raise ValueError(f"problem ID is not safe as a task directory: {problem_id!r}")
    if (
        not resources.image
        or min(resources.cpus, resources.memory_mb, resources.timeout_sec) <= 0
    ):
        raise ValueError("Harbor image and resource limits must be positive")

    task = Path(output) / problem_id
    task.mkdir(parents=True)
    (task / "environment").mkdir()
    tests = task / "tests"
    tests.mkdir()
    (task / "instruction.md").write_text(
        encode_instruction(problem.spec), encoding="utf-8"
    )
    (task / "task.toml").write_text(_config(problem, resources), encoding="utf-8")
    (tests / "template.py").write_text(problem.spec.code_template, encoding="utf-8")
    if problem.answer is not None:
        (tests / "reference.py").write_text(problem.answer.code, encoding="utf-8")
        if problem.answer.testcases is not None:
            (tests / "testcases.json").write_text(
                json.dumps(problem.answer.testcases), encoding="utf-8"
            )
    test_script = tests / "test.sh"
    test_script.write_text(_TEST, encoding="utf-8")
    test_script.chmod(0o755)
    return task


def compile_challenge(
    challenge: Challenge, output: str | Path, resources: Resources
) -> tuple[Path, ...]:
    return tuple(
        compile_problem(problem, output, resources) for problem in challenge.problems
    )


def compile_challenges(
    challenges: list[Challenge], output: str | Path, resources: Resources
) -> tuple[Path, ...]:
    ids = [problem.spec.id for item in challenges for problem in item.problems]
    if len(ids) != len(set(ids)):
        raise ValueError("problem IDs must be unique across challenges")
    return tuple(
        task
        for challenge in challenges
        for task in compile_challenge(challenge, output, resources)
    )
