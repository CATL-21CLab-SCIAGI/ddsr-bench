from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench import __version__
from ddsr_bench.benchmarks.mmmupro.data.loader import OFFICIAL_SETTINGS, iter_directory
from ddsr_bench.benchmarks.mmmupro.data.schemas import AnswerSpec, Problem, ProblemSpec
from ddsr_bench.benchmarks.utils import Resources


def encode_instruction(problem: ProblemSpec) -> str:
    return json.dumps(asdict(problem), indent=2, ensure_ascii=False)


def decode_instruction(instruction: str) -> ProblemSpec:
    data = json.loads(instruction)
    if not isinstance(data, dict):
        raise TypeError("prepared MMMU-Pro instruction must be an object")
    data["choices"] = tuple(data.get("choices", ()))
    data["images"] = tuple(data.get("images", ()))
    problem = ProblemSpec(**data)
    if (
        not problem.id
        or (problem.setting != "vision" and not problem.question.strip())
        or not 2 <= len(problem.choices) <= 26
        or not problem.images
        or any(not image.startswith("data:image/") for image in problem.images)
    ):
        raise ValueError("prepared MMMU-Pro instruction is invalid")
    return problem


def decode_reference(reference: str) -> AnswerSpec:
    data = json.loads(reference)
    if (
        not isinstance(data, dict)
        or data.get("letter") not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ):
        raise ValueError("prepared MMMU-Pro reference is invalid")
    return AnswerSpec(**data)


def _task_key(problem: ProblemSpec) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", problem.id).strip("-")
    return f"{slug}-{sha256(problem.id.encode()).hexdigest()[:8]}"


def compile_problem(problem: Problem, output: str | Path) -> Path:
    if problem.answer is None:
        raise ValueError(f"MMMU-Pro problem {problem.spec.id!r} has no answer")
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
        f'{json.dumps(f"mmmupro/{key}")}\nversion = "{__version__}"\n',
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
        raise ValueError("MMMU-Pro preparation requires its fetched shards")
    settings = tuple(map(str, config.get("official_settings", OFFICIAL_SETTINGS)))
    return tuple(
        compile_problem(problem, output) for problem in iter_directory(source, settings)
    )
