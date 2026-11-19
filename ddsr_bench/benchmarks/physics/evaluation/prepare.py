from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, get_args

from ddsr_bench import __version__
from ddsr_bench.benchmarks.physics.data.loader import (
    DATASET,
    EXPECTED,
    EXPECTED_ANSWERS,
    EXPECTED_FIGURES,
    EXPECTED_TEXT,
    FILES,
    REVISION,
    load_directory,
    load_split,
)
from ddsr_bench.benchmarks.physics.data.schemas import (
    AnswerSpec,
    AnswerType,
    Domain,
    Figure,
    Problem,
    ProblemSpec,
    Split,
)
from ddsr_bench.benchmarks.utils import Resources


def encode_instruction(problem: ProblemSpec) -> str:
    return json.dumps(asdict(problem), indent=2, ensure_ascii=False)


def decode_instruction(instruction: str) -> ProblemSpec:
    data = json.loads(instruction)
    if not isinstance(data, dict):
        raise TypeError("prepared PHYSICS instruction must be an object")
    raw_figures = data.pop("figures", None)
    if not isinstance(raw_figures, list) or any(
        not isinstance(figure, dict)
        or set(figure) != {"url"}
        or not isinstance(figure["url"], str)
        or not figure["url"].startswith("data:image/")
        for figure in raw_figures
    ):
        raise TypeError("prepared PHYSICS instruction has invalid figures")
    spec = ProblemSpec(
        **data, figures=tuple(Figure(**figure) for figure in raw_figures)
    )
    if (
        not isinstance(spec.id, str)
        or not spec.id
        or not isinstance(spec.question, str)
        or not spec.question.strip()
        or spec.split not in get_args(Split)
        or spec.domain not in get_args(Domain)
    ):
        raise ValueError("prepared PHYSICS instruction has invalid public fields")
    return spec


def _reference(answer: AnswerSpec) -> str:
    return json.dumps(asdict(answer), indent=2, ensure_ascii=False)


def _answer(reference: Mapping[str, Any]) -> AnswerSpec:
    fields = ("solution", "final_answers", "labels", "types")
    if any(field not in reference for field in fields):
        raise TypeError("invalid prepared PHYSICS reference")
    solution = reference["solution"]
    values = [reference[field] for field in fields[1:]]
    if not isinstance(solution, str) or not solution.strip():
        raise TypeError("prepared PHYSICS reference requires a solution")
    if any(
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
        for value in values
    ):
        raise TypeError("prepared PHYSICS answer fields must be nonempty text lists")
    answers, labels, types = map(tuple, values)
    if not len(answers) == len(labels) == len(types):
        raise ValueError("prepared PHYSICS answer fields must align")
    if any(answer_type not in get_args(AnswerType) for answer_type in types):
        raise ValueError("prepared PHYSICS reference has an unsupported answer type")
    return AnswerSpec(solution, answers, labels, types)


def decode_reference(reference: str) -> AnswerSpec:
    value = json.loads(reference)
    if not isinstance(value, dict):
        raise TypeError("prepared PHYSICS reference must be an object")
    return _answer(value)


def _task_key(problem: ProblemSpec) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", problem.id).strip("-")
    digest = sha256(problem.id.encode()).hexdigest()[:8]
    return f"{problem.split}-{slug}-{digest}"


def compile_problem(problem: Problem, output: str | Path) -> Path:
    """Write one static PHYSICS task with public and verifier projections."""
    if problem.answer is None:
        raise ValueError(f"PHYSICS problem {problem.spec.id!r} has no reference")
    key = _task_key(problem.spec)
    task = Path(output) / key
    task.mkdir(parents=True)
    tests = task / "tests"
    tests.mkdir()
    (task / "instruction.md").write_text(
        encode_instruction(problem.spec), encoding="utf-8"
    )
    (tests / "reference.json").write_text(_reference(problem.answer), encoding="utf-8")
    (task / "task.toml").write_text(
        f'schema_version = "1.4"\n\n[task]\nname = '
        f'{json.dumps(f"physics/{key}")}\nversion = "{__version__}"\n',
        encoding="utf-8",
    )
    return task


def compile_problems(problems: list[Problem], output: str | Path) -> tuple[Path, ...]:
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("PHYSICS problem IDs must be unique")
    return tuple(compile_problem(problem, output) for problem in problems)


def prepare_tasks(
    config: Mapping[str, Any],
    source: str | Path | None,
    output: str | Path,
    resources: Resources,
) -> tuple[Path, ...]:
    """Compile the pinned verified edition for static/offline evaluation."""
    del resources  # PHYSICS uses host-side LLM judging, not task containers.
    configured = (
        config.get("name"),
        config.get("dataset"),
        config.get("revision"),
        config.get("files"),
        config.get("expected_records"),
        config.get("expected_text_records"),
        config.get("expected_answers"),
        config.get("expected_figures"),
    )
    expected = (
        "physics",
        DATASET,
        REVISION,
        FILES,
        EXPECTED,
        EXPECTED_TEXT,
        EXPECTED_ANSWERS,
        EXPECTED_FIGURES,
    )
    if configured != expected:
        raise ValueError("PHYSICS preparation requires its pinned configuration")
    problems = (
        load_directory(source)
        if source is not None
        else [
            problem
            for split in FILES
            for problem in load_split(split, revision=REVISION)
        ]
    )
    return compile_problems(problems, output)
