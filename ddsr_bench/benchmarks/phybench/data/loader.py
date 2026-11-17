from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.utils import benchmark_config, read_jsonl, read_text

from .schemas import AnswerSpec, Problem, ProblemSpec

_CONFIG = benchmark_config("phybench")
DATASET = str(_CONFIG["dataset"])
REVISION = str(_CONFIG["revision"])
DATA_FILE = str(_CONFIG["data_file"])
SPLIT = str(_CONFIG["split"])
GRADABLE_ONLY = _CONFIG["gradable_only"] is True
EXPECTED_RECORDS = int(_CONFIG["expected_records"])


def _text(record: Mapping[str, Any], problem_id: str, *keys: str) -> str:
    for key in keys:
        if key in record:
            return read_text(record, key, problem_id)
    raise TypeError(f"{problem_id} requires text at one of {keys!r}")


def load_problem(record: Mapping[str, Any]) -> Problem:
    """Normalize one official or LOCA-corrected PHYBench record."""
    raw_id = record.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, int | str):
        raise TypeError("PHYBench problem requires an integer or text ID")
    problem_id = str(raw_id)
    solution = _text(record, problem_id, "solutions", "solution")
    value = _text(record, problem_id, "final_answers", "answer")
    if bool(solution.strip()) != bool(value.strip()):
        raise ValueError(f"{problem_id} must provide both solution and answer")

    spec = ProblemSpec(
        id=problem_id,
        tag=read_text(record, "tag", problem_id),
        content=_text(record, problem_id, "questions", "content"),
    )
    answer = AnswerSpec(value=value, solution=solution) if value.strip() else None
    return Problem(spec=spec, answer=answer)


def load_problems(
    records: Iterable[Mapping[str, Any]], gradable_only: bool = GRADABLE_ONLY
) -> list[Problem]:
    """Normalize records and optionally omit records without references."""
    problems = [load_problem(record) for record in records]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("PHYBench problem IDs must be unique")
    if gradable_only:
        return [problem for problem in problems if problem.answer is not None]
    return problems


def load_split(
    split: str = SPLIT,
    revision: str = REVISION,
    gradable_only: bool = GRADABLE_ONLY,
) -> list[Problem]:
    """Load the pinned corrected set through the datasets download cache."""
    from datasets import load_dataset

    data_file = f"https://raw.githubusercontent.com/{DATASET}/{revision}/{DATA_FILE}"
    records: Iterable[Mapping[str, Any]] = load_dataset(
        "json",
        data_files=data_file,
        split=split,
    )
    return _checked(load_problems(records, gradable_only))


def load_file(path: str | Path, gradable_only: bool = GRADABLE_ONLY) -> list[Problem]:
    """Load the pinned PHYBench edition from a local JSONL file."""
    return _checked(load_problems(read_jsonl(path), gradable_only))


def validate_source(path: Path) -> None:
    """Validate one materialized PHYBench edition."""
    load_file(path / Path(DATA_FILE).name, gradable_only=False)


def _checked(problems: list[Problem]) -> list[Problem]:
    if len(problems) != EXPECTED_RECORDS:
        raise ValueError("PHYBench does not match its pinned release count")
    return problems
