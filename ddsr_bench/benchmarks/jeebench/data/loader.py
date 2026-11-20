from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

from ddsr_bench.benchmarks.utils import benchmark_config, read_text

from .schemas import AnswerSpec, Problem, ProblemSpec, ResponseType

_CONFIG = benchmark_config("jeebench")
_TYPES = {"MCQ", "MCQ(multiple)", "Integer", "Numeric"}
_EXPECTED_TYPES = {
    str(name): int(size) for name, size in _CONFIG["expected_response_types"].items()
}
_EXPECTED_SUBJECTS = {
    str(name): int(size) for name, size in _CONFIG["expected_subjects"].items()
}


def _id(paper: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", paper.casefold()).strip("-")
    return f"{slug}-{index}"


def load_problem(record: Mapping[str, Any]) -> Problem:
    """Normalize one official JEEBench row and keep its answer private."""
    paper = read_text(record, "description", "JEEBench")
    index = record.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise TypeError(f"{paper} requires a positive integer index")
    problem_id = _id(paper, index)
    response_type = cast(ResponseType, read_text(record, "type", problem_id))
    if response_type not in _TYPES:
        raise ValueError(f"{problem_id} has unknown response type {response_type!r}")
    question = read_text(record, "question", problem_id)
    answer = read_text(record, "gold", problem_id)
    return Problem(
        ProblemSpec(
            id=problem_id,
            paper=paper,
            index=index,
            subject=read_text(record, "subject", problem_id),
            response_type=response_type,
            question=question.replace("\n\n", "\n").strip(),
        ),
        AnswerSpec(answer),
    )


def _read(archive: ZipFile, filename: str) -> Any:
    return json.loads(archive.read(filename))


def load_directory(path: str | Path) -> list[Problem]:
    """Load the pinned official dataset archive."""
    archive_path = Path(path) / str(_CONFIG["data_file"])
    with ZipFile(archive_path) as archive:
        records = _read(archive, "data/dataset.json")
    if not isinstance(records, list) or any(
        not isinstance(x, Mapping) for x in records
    ):
        raise TypeError("JEEBench dataset must be a list of objects")
    problems = [load_problem(record) for record in records]
    if len(problems) != int(_CONFIG["expected_records"]):
        raise ValueError("JEEBench does not match its pinned size")
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("JEEBench has duplicate problem IDs")
    types = Counter(problem.spec.response_type for problem in problems)
    subjects = Counter(problem.spec.subject for problem in problems)
    if types != _EXPECTED_TYPES or subjects != _EXPECTED_SUBJECTS:
        raise ValueError("JEEBench does not match its pinned composition")
    return problems


def load_examples(path: str | Path) -> dict[str, dict[str, dict[str, str]]]:
    """Load the official one-shot demonstrations from the archive."""
    archive_path = Path(path) / str(_CONFIG["data_file"])
    with ZipFile(archive_path) as archive:
        examples = _read(archive, "data/few_shot_examples.json")
    if not isinstance(examples, dict):
        raise TypeError("JEEBench examples must be an object")
    return examples


def validate_source(path: Path) -> None:
    load_directory(path)
    examples = load_examples(path)
    if set(examples) != set(_EXPECTED_SUBJECTS) or any(
        set(subject) != _TYPES for subject in examples.values()
    ):
        raise ValueError("JEEBench examples do not cover every subject and type")
