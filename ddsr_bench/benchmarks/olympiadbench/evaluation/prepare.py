from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, get_args

from ddsr_bench import __version__
from ddsr_bench.benchmarks.olympiadbench.data.loader import (
    DATASET,
    EXPECTED,
    FILES,
    REVISION,
    load_directory,
    load_split,
)
from ddsr_bench.benchmarks.olympiadbench.data.schemas import (
    AnswerSpec,
    Kind,
    Language,
    Modality,
    Problem,
    ProblemSpec,
    Source,
    Subject,
)
from ddsr_bench.benchmarks.utils import Resources


def encode_instruction(problem: ProblemSpec) -> str:
    return json.dumps(asdict(problem), indent=2, ensure_ascii=False)


def decode_instruction(instruction: str) -> ProblemSpec:
    data = json.loads(instruction)
    if not isinstance(data, dict):
        raise TypeError("prepared OlympiadBench instruction must be an object")
    data["images"] = tuple(data.get("images", ()))
    problem = ProblemSpec(**data)
    if (
        not problem.id
        or not problem.question.strip()
        or problem.kind not in get_args(Kind)
        or problem.modality not in get_args(Modality)
        or problem.subject not in get_args(Subject)
        or problem.language not in get_args(Language)
        or problem.source not in get_args(Source)
        or any(not image.startswith("data:image/") for image in problem.images)
    ):
        raise ValueError("prepared OlympiadBench instruction is invalid")
    return problem


def decode_reference(reference: str) -> AnswerSpec:
    data = json.loads(reference)
    if not isinstance(data, dict):
        raise TypeError("prepared OlympiadBench reference must be an object")
    data["solutions"] = tuple(data.get("solutions", ()))
    if data.get("answers") is not None:
        data["answers"] = tuple(data["answers"])
    return AnswerSpec(**data)


def _task_key(problem: ProblemSpec) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", problem.id).strip("-")
    return f"{slug}-{sha256(problem.id.encode()).hexdigest()[:8]}"


def compile_problem(problem: Problem, output: str | Path) -> Path:
    if problem.answer is None:
        raise ValueError(f"OlympiadBench problem {problem.spec.id!r} has no answer")
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
        f'{json.dumps(f"olympiadbench/{key}")}\nversion = "{__version__}"\n',
        encoding="utf-8",
    )
    return task


def compile_problems(problems: list[Problem], output: str | Path) -> tuple[Path, ...]:
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("OlympiadBench problem IDs must be unique")
    return tuple(compile_problem(problem, output) for problem in problems)


def prepare_tasks(
    config: Mapping[str, Any],
    source: str | Path | None,
    output: str | Path,
    resources: Resources,
) -> tuple[Path, ...]:
    del resources
    configured = (
        config.get("name"),
        config.get("dataset"),
        config.get("revision"),
        tuple(config.get("files", ())),
        dict(config.get("expected_records", {})),
    )
    if configured != ("olympiadbench", DATASET, REVISION, FILES, EXPECTED):
        raise ValueError("OlympiadBench preparation requires its pinned config")
    problems = load_directory(source) if source is not None else load_split()
    return compile_problems(problems, output)
