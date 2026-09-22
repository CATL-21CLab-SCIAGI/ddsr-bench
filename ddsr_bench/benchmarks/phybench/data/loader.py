from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ddsr_bench.benchmarks.utils import read_text

from .schemas import AnswerSpec, Problem, ProblemSpec

DATASET = "Eureka-Lab/PHYBench"
REVISION = "d6d91c787b7abb865eb2490a328bf85a9f5095f0"
DATA_FILE = "PHYBench-questions_v1.json"


def load_problem(record: Mapping[str, Any]) -> Problem:
    """Normalize one record from the official PHYBench dataset."""
    raw_id = record.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, int | str):
        raise TypeError("PHYBench problem requires an integer or text ID")
    problem_id = str(raw_id)
    solution = read_text(record, "solution", problem_id)
    value = read_text(record, "answer", problem_id)
    if bool(solution.strip()) != bool(value.strip()):
        raise ValueError(f"{problem_id} must provide both solution and answer")

    spec = ProblemSpec(
        id=problem_id,
        tag=read_text(record, "tag", problem_id),
        content=read_text(record, "content", problem_id),
    )
    answer = AnswerSpec(value=value, solution=solution) if value.strip() else None
    return Problem(spec=spec, answer=answer)


def load_problems(
    records: Iterable[Mapping[str, Any]], gradable_only: bool = True
) -> list[Problem]:
    """Normalize records and optionally omit the 400 public-only problems."""
    problems = [load_problem(record) for record in records]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("PHYBench problem IDs must be unique")
    if gradable_only:
        return [problem for problem in problems if problem.answer is not None]
    return problems


def load_split(
    split: str = "train",
    revision: str = REVISION,
    gradable_only: bool = True,
) -> list[Problem]:
    """Load the canonical 500-row file without concatenating its subsets."""
    from datasets import load_dataset

    records: Iterable[Mapping[str, Any]] = load_dataset(
        DATASET,
        data_files=DATA_FILE,
        split=split,
        revision=revision,
    )
    return load_problems(records, gradable_only)
