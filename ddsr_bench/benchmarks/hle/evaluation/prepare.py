from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, get_args

from ddsr_bench import __version__
from ddsr_bench.benchmarks.hle.data.loader import (
    DATASET,
    EXPECTED,
    EXPECTED_IMAGES,
    EXPECTED_RATIONALE_IMAGES,
    EXPECTED_SELECTED,
    EXPECTED_SELECTED_IMAGES,
    EXPECTED_SELECTED_RATIONALE_IMAGES,
    EXPECTED_SELECTED_TYPES,
    EXPECTED_TYPES,
    FILES,
    REVISION,
    SUBSETS,
    load_directory,
    load_split,
    select,
)
from ddsr_bench.benchmarks.hle.data.schemas import (
    AnswerSpec,
    AnswerType,
    Problem,
    ProblemSpec,
    Subset,
)
from ddsr_bench.benchmarks.utils import Resources


def encode_instruction(problem: ProblemSpec) -> str:
    return json.dumps(asdict(problem), indent=2, ensure_ascii=False)


def decode_instruction(instruction: str) -> ProblemSpec:
    data = json.loads(instruction)
    if not isinstance(data, dict):
        raise TypeError("prepared HLE instruction must be an object")
    spec = ProblemSpec(**data)
    if (
        any(
            not isinstance(value, str)
            for value in (
                spec.id,
                spec.subset,
                spec.category,
                spec.subject,
                spec.question,
            )
        )
        or not spec.id
        or not spec.question.strip()
        or spec.subset not in get_args(Subset)
        or not spec.category.strip()
        or not spec.subject.strip()
        or (
            spec.image is not None
            and (
                not isinstance(spec.image, str)
                or not spec.image.startswith("data:image/")
            )
        )
    ):
        raise ValueError("prepared HLE instruction has invalid public fields")
    return spec


def decode_reference(reference: str) -> AnswerSpec:
    data = json.loads(reference)
    if not isinstance(data, dict):
        raise TypeError("prepared HLE reference must be an object")
    answer = AnswerSpec(**data)
    if (
        not all(
            isinstance(value, str)
            for value in (answer.answer, answer.type, answer.rationale)
        )
        or not answer.answer.strip()
        or answer.type not in get_args(AnswerType)
        or (
            answer.rationale_image is not None
            and (
                not isinstance(answer.rationale_image, str)
                or not answer.rationale_image.startswith("data:image/")
            )
        )
    ):
        raise ValueError("prepared HLE reference has invalid fields")
    return answer


def _task_key(problem: ProblemSpec) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", problem.id).strip("-")
    digest = sha256(problem.id.encode()).hexdigest()[:8]
    return f"{problem.subset.lower()}-{slug}-{digest}"


def compile_problem(problem: Problem, output: str | Path) -> Path:
    """Write one static HLE task with public and verifier projections."""
    if problem.answer is None:
        raise ValueError(f"HLE problem {problem.spec.id!r} has no reference")
    key = _task_key(problem.spec)
    task = Path(output) / key
    task.mkdir(parents=True)
    tests = task / "tests"
    tests.mkdir()
    (task / "instruction.md").write_text(
        encode_instruction(problem.spec), encoding="utf-8"
    )
    (tests / "reference.json").write_text(
        json.dumps(asdict(problem.answer), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (task / "task.toml").write_text(
        f'schema_version = "1.4"\n\n[task]\nname = '
        f'{json.dumps(f"hle/{key}")}\nversion = "{__version__}"\n',
        encoding="utf-8",
    )
    return task


def compile_problems(problems: list[Problem], output: str | Path) -> tuple[Path, ...]:
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("HLE problem IDs must be unique")
    return tuple(compile_problem(problem, output) for problem in problems)


def prepare_tasks(
    config: Mapping[str, Any],
    source: str | Path | None,
    output: str | Path,
    resources: Resources,
) -> tuple[Path, ...]:
    """Compile the pinned certified HLE-Verified selection for offline use."""
    del resources
    configured = (
        config.get("name"),
        config.get("dataset"),
        config.get("revision"),
        tuple(config.get("files", ())),
        config.get("expected_records"),
        config.get("expected_answer_types"),
        config.get("expected_images"),
        config.get("expected_rationale_images"),
        config.get("expected_selected_records"),
        config.get("expected_selected_answer_types"),
        config.get("expected_selected_images"),
        config.get("expected_selected_rationale_images"),
        tuple(config.get("subsets", ())),
    )
    expected = (
        "hle",
        DATASET,
        REVISION,
        FILES,
        EXPECTED,
        EXPECTED_TYPES,
        EXPECTED_IMAGES,
        EXPECTED_RATIONALE_IMAGES,
        EXPECTED_SELECTED,
        EXPECTED_SELECTED_TYPES,
        EXPECTED_SELECTED_IMAGES,
        EXPECTED_SELECTED_RATIONALE_IMAGES,
        SUBSETS,
    )
    if configured != expected:
        raise ValueError("HLE preparation requires its pinned configuration")
    problems = select(load_directory(source)) if source is not None else load_split()
    return compile_problems(problems, output)
