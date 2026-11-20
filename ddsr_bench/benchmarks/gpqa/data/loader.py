from __future__ import annotations

import csv
import io
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from ddsr_bench.benchmarks.utils import benchmark_config, read_text

from .schemas import AnswerSpec, Problem, ProblemSpec

_CONFIG = benchmark_config("gpqa")
COHORTS = tuple(str(name) for name in _CONFIG["expected_records"])
EXPECTED = {str(name): int(size) for name, size in _CONFIG["expected_records"].items()}
PASSWORD = str(_CONFIG["archive_password"]).encode()


def load_problem(record: Mapping[str, Any], cohort: str, rng: random.Random) -> Problem:
    """Normalize one row with the official deterministic choice shuffle."""
    problem_id = read_text(record, "Record ID", cohort)
    correct = read_text(record, "Correct Answer", problem_id)
    choices = [
        read_text(record, f"Incorrect Answer {index}", problem_id)
        for index in range(1, 4)
    ] + [correct]
    rng.shuffle(choices)
    answer = AnswerSpec(
        letter="ABCD"[choices.index(correct)],
        explanation=read_text(record, "Explanation", problem_id),
    )
    spec = ProblemSpec(
        id=problem_id,
        cohort=cohort,
        domain=read_text(record, "High-level domain", problem_id),
        subdomain=read_text(record, "Subdomain", problem_id),
        question=read_text(record, "Question", problem_id),
        choices=tuple(choices),
    )
    return Problem(spec, answer)


def _records(archive: ZipFile, cohort: str) -> list[dict[str, str]]:
    filename = f"dataset/gpqa_{cohort}.csv"
    with archive.open(filename, pwd=PASSWORD) as source:
        text = io.TextIOWrapper(source, encoding="utf-8-sig", newline="")
        return list(csv.DictReader(text))


def load_directory(path: str | Path, cohort: str = "main") -> list[Problem]:
    """Load one official cohort from the pinned password-protected archive."""
    if cohort not in COHORTS:
        raise ValueError(f"unknown GPQA cohort {cohort!r}")
    with ZipFile(Path(path) / str(_CONFIG["data_file"])) as archive:
        records = _records(archive, cohort)
    if len(records) != EXPECTED[cohort]:
        raise ValueError(f"GPQA {cohort!r} does not match its pinned size")
    rng = random.Random(int(_CONFIG["choice_seed"]))
    problems = [load_problem(record, cohort, rng) for record in records]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError(f"GPQA {cohort!r} has duplicate record IDs")
    return problems


def validate_source(path: Path) -> None:
    for cohort in COHORTS:
        load_directory(path, cohort)
    with ZipFile(Path(path) / str(_CONFIG["data_file"])) as archive:
        experts = _records(archive, "experts")
    if len(experts) != int(_CONFIG["expected_experts"]):
        raise ValueError("GPQA expert metadata does not match its pinned size")
