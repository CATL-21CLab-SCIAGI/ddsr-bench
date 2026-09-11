from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, get_args

from .schemas import AnswerSpec, AnswerType, Problem, ProblemSpec

DATASET = "weidawang/CMPhysBench"
REVISION = "43d185851f731e23aa5737c3667b3a9e87bf8cd1"


def _text(record: Mapping[str, Any], key: str, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{label} requires text at {key!r}")
    return value


def load_problem(record: Mapping[str, Any]) -> Problem:
    """Normalize one record from the official Hugging Face dataset."""
    raw_id = record.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, int | str):
        raise TypeError("CMPhysBench problem requires an integer or text ID")
    problem_id = str(raw_id)
    answer_type = _text(record, "answer_type", problem_id)
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
        question=_text(record, "question", problem_id),
        symbols=_text(record, "symbol", problem_id),
        answer_type=answer_type,
        topic=_text(record, "topic", problem_id),
    )
    answer = AnswerSpec(final_answers[0]) if final_answers else None
    return Problem(spec=spec, answer=answer)


def load_problems(
    records: Iterable[Mapping[str, Any]], gradable_only: bool = True
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
    split: str = "train",
    revision: str = REVISION,
    gradable_only: bool = True,
) -> list[Problem]:
    """Load one pinned CMPhysBench split from Hugging Face."""
    from datasets import load_dataset

    records: Iterable[Mapping[str, Any]] = load_dataset(
        DATASET, split=split, revision=revision
    )
    return load_problems(records, gradable_only)
