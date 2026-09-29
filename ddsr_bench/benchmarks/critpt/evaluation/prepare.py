from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ddsr_bench import __version__
from ddsr_bench.benchmarks.critpt.data.loader import load_ai_problems, load_challenges
from ddsr_bench.benchmarks.critpt.data.schemas import (
    AIProblem,
    AIProblemSpec,
    Challenge,
    CritPtSource,
    Problem,
    ProblemSpec,
)
from ddsr_bench.benchmarks.utils import Resources, write_json

_TEST = """#!/bin/sh
set -eu
python -m ddsr_bench.benchmarks.critpt.evaluation.verifier
"""


def encode_instruction(problem: ProblemSpec | AIProblemSpec) -> str:
    """Serialize public fields as JSON for Harbor's required instruction.md."""
    data = asdict(problem)
    data.pop("source_path")
    return json.dumps(data, indent=2, ensure_ascii=False)


def decode_instruction(instruction: str) -> ProblemSpec:
    """Restore the public problem fields from one prepared task."""
    data = json.loads(instruction)
    data["source"] = CritPtSource(data.get("source"))
    spec = AIProblemSpec if data["source"] == CritPtSource.AI else ProblemSpec
    return spec(**data, source_path=Path("prepared-task"))


def _config(problem: Problem | AIProblem, resources: Resources) -> str:
    name = json.dumps(f"critpt/{problem.spec.id}")
    image = json.dumps(resources.image)
    artifacts = (
        "[]"
        if isinstance(problem, AIProblem)
        else '[{ source = "/app/answer.py", destination = "answer.py" }]'
    )
    return f"""schema_version = "1.4"
artifacts = {artifacts}

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
    problem: Problem | AIProblem,
    output: str | Path,
    resources: Resources,
    *,
    source_root: Path | None = None,
) -> Path:
    """Compile one CritPt problem into one Harbor task directory."""
    problem_id = problem.spec.id
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", problem_id) is None:
        raise ValueError(f"problem ID is not safe as a task directory: {problem_id!r}")
    if (
        not resources.image
        or min(resources.cpus, resources.memory_mb, resources.timeout_sec) <= 0
    ):
        raise ValueError("Harbor image and resource limits must be positive")

    if isinstance(problem, AIProblem):
        source = problem.spec.source_path.resolve()
        # The corpus marker keeps single-file and directory preparation consistent.
        root = next(
            (p for p in source.parents if (p / "CORPUS_INDEX.json").is_file()),
            source_root.resolve() if source_root is not None else source.parent,
        )
        reference = {k: v for k, v in asdict(problem.answer).items() if v is not None}
        reference["source_path"] = source.relative_to(root).as_posix()

    task = Path(output) / problem_id
    task.mkdir(parents=True)
    (task / "environment").mkdir()
    tests = task / "tests"
    tests.mkdir()
    (task / "instruction.md").write_text(
        encode_instruction(problem.spec), encoding="utf-8"
    )
    (task / "task.toml").write_text(_config(problem, resources), encoding="utf-8")
    script = _TEST
    if isinstance(problem, AIProblem):
        write_json(tests / "reference.json", reference, overwrite=False)
        if problem.answer.code:
            (tests / "reference.py").write_text(problem.answer.code, encoding="utf-8")
        script = '#!/bin/sh\necho "AI task verification requires a customized grader" >&2\nexit 2\n'
    else:
        (tests / "template.py").write_text(problem.spec.code_template, encoding="utf-8")
    if isinstance(problem, Problem) and problem.answer is not None:
        (tests / "reference.py").write_text(problem.answer.code, encoding="utf-8")
        if problem.answer.testcases is not None:
            (tests / "testcases.json").write_text(
                json.dumps(problem.answer.testcases), encoding="utf-8"
            )
    test_script = tests / "test.sh"
    test_script.write_text(script, encoding="utf-8")
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


def prepare_tasks(
    config: Mapping[str, Any],
    source: str | Path | None,
    output: str | Path,
    resources: Resources,
) -> tuple[Path, ...]:
    """Load CritPt source data and compile its Harbor tasks."""
    if config.get("name") != "critpt":
        raise ValueError("CritPt preparation requires its benchmark configuration")
    if source is None:
        raise ValueError("paths.input is required for CritPt preparation")
    origin = CritPtSource(config.get("source", CritPtSource.OFFICIAL))
    if origin == CritPtSource.AI:
        root = Path(source)
        if root.is_file():
            root = root.parent
        return tuple(
            compile_problem(p, output, resources, source_root=root)
            for p in load_ai_problems(source)
        )
    return compile_challenges(load_challenges(source), output, resources)
