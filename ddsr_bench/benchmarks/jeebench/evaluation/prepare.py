from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench import __version__
from ddsr_bench.benchmarks.jeebench.data.loader import load_directory
from ddsr_bench.benchmarks.jeebench.data.schemas import AnswerSpec, Problem, ProblemSpec
from ddsr_bench.benchmarks.utils import Resources


def encode_instruction(problem: ProblemSpec) -> str:
    return json.dumps(asdict(problem), indent=2, ensure_ascii=False)


def decode_instruction(instruction: str) -> ProblemSpec:
    data = json.loads(instruction)
    if not isinstance(data, dict):
        raise TypeError("prepared JEEBench instruction must be an object")
    problem = ProblemSpec(**data)
    if not problem.id or not problem.question.strip() or problem.index < 1:
        raise ValueError("prepared JEEBench instruction is invalid")
    return problem


def decode_reference(reference: str) -> AnswerSpec:
    data = json.loads(reference)
    if not isinstance(data, dict) or not isinstance(data.get("value"), str):
        raise TypeError("prepared JEEBench reference is invalid")
    return AnswerSpec(**data)


def _task_key(problem: ProblemSpec) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", problem.id).strip("-")
    return f"{slug}-{sha256(problem.id.encode()).hexdigest()[:8]}"


def compile_problem(problem: Problem, output: str | Path) -> Path:
    if problem.answer is None:
        raise ValueError(f"JEEBench problem {problem.spec.id!r} has no answer")
    key = _task_key(problem.spec)
    task = Path(output) / key
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
        f'{json.dumps(f"jeebench/{key}")}\nversion = "{__version__}"\n',
        encoding="utf-8",
    )
    return task


def prepare_tasks(
    config: Mapping[str, Any],
    source: str | Path | None,
    output: str | Path,
    resources: Resources,
) -> tuple[Path, ...]:
    del resources
    if source is None:
        raise ValueError("JEEBench preparation requires its fetched archive")
    problems = load_directory(source)
    return tuple(compile_problem(problem, output) for problem in problems)
