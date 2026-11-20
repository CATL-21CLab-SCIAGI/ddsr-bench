from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, get_args

from ddsr_bench import __version__
from ddsr_bench.benchmarks.frontierscience.data.loader import (
    DATASET,
    EXPECTED,
    EXPECTED_SUBJECTS,
    FILES,
    REVISION,
    load_directory,
    load_split,
)
from ddsr_bench.benchmarks.frontierscience.data.schemas import (
    AnswerSpec,
    Problem,
    ProblemSpec,
    Subject,
    Track,
)
from ddsr_bench.benchmarks.utils import Resources


def encode_instruction(problem: ProblemSpec) -> str:
    return json.dumps(asdict(problem), indent=2, ensure_ascii=False)


def decode_instruction(instruction: str) -> ProblemSpec:
    data = json.loads(instruction)
    if not isinstance(data, dict):
        raise TypeError("prepared FrontierScience instruction must be an object")
    problem = ProblemSpec(**data)
    if (
        not problem.id
        or not problem.prompt.strip()
        or problem.track not in get_args(Track)
        or problem.subject not in get_args(Subject)
    ):
        raise ValueError("prepared FrontierScience instruction is invalid")
    return problem


def decode_reference(reference: str) -> AnswerSpec:
    data = json.loads(reference)
    if not isinstance(data, dict):
        raise TypeError("prepared FrontierScience reference must be an object")
    answer = AnswerSpec(**data)
    if not isinstance(answer.reference, str) or not answer.reference.strip():
        raise ValueError("prepared FrontierScience reference is invalid")
    return answer


def _task_key(problem: ProblemSpec) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", problem.id).strip("-")
    return f"{problem.track}-{slug}-{sha256(problem.id.encode()).hexdigest()[:8]}"


def compile_problem(problem: Problem, output: str | Path) -> Path:
    """Write public input and verifier-only reference projections."""
    if problem.answer is None:
        raise ValueError(
            f"FrontierScience problem {problem.spec.id!r} has no reference"
        )
    task = Path(output) / _task_key(problem.spec)
    tests = task / "tests"
    tests.mkdir(parents=True)
    (task / "instruction.md").write_text(
        encode_instruction(problem.spec), encoding="utf-8"
    )
    (tests / "reference.json").write_text(
        json.dumps(asdict(problem.answer), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (task / "task.toml").write_text(
        f'schema_version = "1.4"\n\n[task]\nname = '
        f'{json.dumps(f"frontierscience/{_task_key(problem.spec)}")}\n'
        f'version = "{__version__}"\n',
        encoding="utf-8",
    )
    return task


def compile_problems(problems: list[Problem], output: str | Path) -> tuple[Path, ...]:
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("FrontierScience problem IDs must be unique")
    return tuple(compile_problem(problem, output) for problem in problems)


def prepare_tasks(
    config: Mapping[str, Any],
    source: str | Path | None,
    output: str | Path,
    resources: Resources,
) -> tuple[Path, ...]:
    """Compile the complete pinned gold set for offline fallback."""
    del resources
    configured = (
        config.get("name"),
        config.get("dataset"),
        config.get("revision"),
        tuple(config.get("files", ())),
        dict(config.get("expected_records", {})),
        dict(config.get("expected_subjects", {})),
    )
    expected = (
        "frontierscience",
        DATASET,
        REVISION,
        FILES,
        EXPECTED,
        EXPECTED_SUBJECTS,
    )
    if configured != expected:
        raise ValueError("FrontierScience preparation requires its pinned config")
    problems = load_directory(source) if source is not None else load_split()
    return compile_problems(problems, output)
