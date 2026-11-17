from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, get_args

from ddsr_bench.benchmarks.utils import benchmark_config, read_text

from .schemas import AnswerSpec, AnswerType, Problem, ProblemSpec

_CONFIG = benchmark_config("cmphysbench")
DATASET = str(_CONFIG["dataset"])
REVISION = str(_CONFIG["revision"])
SPLIT = str(_CONFIG["split"])
GRADABLE_ONLY = _CONFIG["gradable_only"] is True
EXPECTED_RECORDS = int(_CONFIG["expected_records"])
EXPECTED_GRADABLE = int(_CONFIG["expected_gradable"])


def load_problem(record: Mapping[str, Any]) -> Problem:
    """Normalize one record from the official Hugging Face dataset."""
    raw_id = record.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, int | str):
        raise TypeError("CMPhysBench problem requires an integer or text ID")
    problem_id = str(raw_id)
    answer_type = read_text(record, "answer_type", problem_id)
    if answer_type not in get_args(AnswerType):
        raise ValueError(f"unsupported CMPhysBench answer type {answer_type!r}")

    context = record.get("context", "")
    if not isinstance(context, str):
        raise TypeError(f"{problem_id} requires optional text at 'context'")
    final_answers = record.get("final_answer")
    if not isinstance(final_answers, list) or any(
        not isinstance(answer, str) for answer in final_answers
    ):
        raise TypeError(f"{problem_id} requires a list of final answers")

    spec = ProblemSpec(
        id=problem_id,
        context=context,
        question=read_text(record, "question", problem_id),
        symbols=read_text(record, "symbol", problem_id),
        answer_type=answer_type,
        topic=read_text(record, "topic", problem_id),
    )
    answer = AnswerSpec(final_answers[0]) if final_answers else None
    return Problem(spec=spec, answer=answer)


def load_problems(
    records: Iterable[Mapping[str, Any]], gradable_only: bool = GRADABLE_ONLY
) -> list[Problem]:
    """Normalize a dataset and optionally omit records without references."""
    problems = [load_problem(record) for record in records]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("CMPhysBench problem IDs must be unique")
    if gradable_only:
        return [problem for problem in problems if problem.answer is not None]
    return problems


def load_split(
    split: str = SPLIT,
    revision: str = REVISION,
    gradable_only: bool = GRADABLE_ONLY,
) -> list[Problem]:
    """Load one pinned CMPhysBench split from Hugging Face."""
    from datasets import load_dataset

    records: Iterable[Mapping[str, Any]] = load_dataset(
        DATASET, split=split, revision=revision
    )
    return _checked(load_problems(records, gradable_only=False), gradable_only)


def load_file(path: str | Path, gradable_only: bool = GRADABLE_ONLY) -> list[Problem]:
    """Load the pinned CMPhysBench edition from a local JSON file."""
    source = Path(path)
    try:
        records = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {source}") from error
    if not isinstance(records, list) or any(
        not isinstance(record, Mapping) for record in records
    ):
        raise TypeError(f"{source} must contain a JSON list of objects")
    return _checked(load_problems(records, gradable_only=False), gradable_only)


def validate_source(path: Path) -> None:
    """Validate one materialized CMPhysBench edition."""
    load_file(path / str(_CONFIG["data_file"]), gradable_only=False)


def _checked(problems: list[Problem], gradable_only: bool) -> list[Problem]:
    if len(problems) != EXPECTED_RECORDS:
        raise ValueError("CMPhysBench does not match its pinned release count")
    gradable = [problem for problem in problems if problem.answer is not None]
    if len(gradable) != EXPECTED_GRADABLE:
        raise ValueError("CMPhysBench does not match its pinned gradable count")
    return gradable if gradable_only else problems
