from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, get_args

from ddsr_bench.benchmarks.utils import benchmark_config, read_jsonl, read_text

from .schemas import AnswerSpec, AnswerType, Domain, Figure, Problem, ProblemSpec, Split

_CONFIG = benchmark_config("physics")
DATASET = str(_CONFIG["dataset"])
REVISION = str(_CONFIG["revision"])
FILES = {str(key): str(value) for key, value in _CONFIG["files"].items()}
EXPECTED = {str(key): int(value) for key, value in _CONFIG["expected_records"].items()}
EXPECTED_TEXT = {
    str(key): int(value) for key, value in _CONFIG["expected_text_records"].items()
}
EXPECTED_FIGURES = {
    str(key): int(value) for key, value in _CONFIG["expected_figures"].items()
}
EXPECTED_ANSWERS = {
    str(key): int(value) for key, value in _CONFIG["expected_answers"].items()
}


def _texts(record: Mapping[str, Any], key: str, problem_id: str) -> tuple[str, ...]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise TypeError(f"{problem_id} requires a nonempty list at {key!r}")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{problem_id} requires nonempty text at {key!r}")
    return tuple(value)


def _figures(value: Any, problem_id: str) -> tuple[Figure, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{problem_id} requires a list at 'figures'")
    figures = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"{problem_id} has an invalid figure")
        image = item.get("image_url")
        url = image.get("url") if isinstance(image, Mapping) else None
        if item.get("type") != "image_url" or not isinstance(url, str):
            raise TypeError(f"{problem_id} has an invalid figure")
        if not url.startswith("data:image/"):
            raise ValueError(f"{problem_id} figure must be an image data URI")
        figures.append(Figure(url))
    return tuple(figures)


def load_problem(record: Mapping[str, Any], split: Split) -> Problem:
    """Normalize one official row and separate public from verifier data."""
    if split not in get_args(Split):
        raise ValueError(f"unknown PHYSICS split {split!r}")
    problem_id = read_text(record, "id", "PHYSICS problem")
    domain = read_text(record, "domain", problem_id)
    if domain not in get_args(Domain):
        raise ValueError(f"{problem_id} has an unsupported domain {domain!r}")
    question = read_text(record, "questions", problem_id)
    solution = read_text(record, "solutions", problem_id)
    if not question.strip() or not solution.strip():
        raise ValueError(f"{problem_id} requires a question and solution")

    answers = _texts(record, "final_answers", problem_id)
    labels = _texts(record, "answer_labels", problem_id)
    answer_types = _texts(record, "answer_types", problem_id)
    if not len(answers) == len(labels) == len(answer_types):
        raise ValueError(f"{problem_id} has misaligned answers")
    if any(answer_type not in get_args(AnswerType) for answer_type in answer_types):
        raise ValueError(f"{problem_id} has an unsupported answer type")

    spec = ProblemSpec(
        id=problem_id,
        split=split,
        domain=domain,
        question=question,
        figures=_figures(record.get("figures"), problem_id),
    )
    answer = AnswerSpec(solution, answers, labels, answer_types)
    return Problem(spec, answer)


def load_problems(records: Iterable[Mapping[str, Any]], split: Split) -> list[Problem]:
    problems = [load_problem(record, split) for record in records]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError(f"PHYSICS {split} problem IDs must be unique")
    if len(problems) != EXPECTED[split]:
        raise ValueError(f"PHYSICS {split} does not match its pinned record count")
    if sum(not problem.spec.figures for problem in problems) != EXPECTED_TEXT[split]:
        raise ValueError(f"PHYSICS {split} does not match its pinned text-only count")
    if (
        sum(len(problem.spec.figures) for problem in problems)
        != EXPECTED_FIGURES[split]
    ):
        raise ValueError(f"PHYSICS {split} does not match its pinned figure count")
    if (
        sum(len(problem.answer.final_answers) for problem in problems)
        != EXPECTED_ANSWERS[split]
    ):
        raise ValueError(f"PHYSICS {split} does not match its pinned answer count")
    return problems


def load_file(path: str | Path, split: Split) -> list[Problem]:
    return load_problems(read_jsonl(path), split)


def load_directory(path: str | Path) -> list[Problem]:
    root = Path(path)
    problems = [
        problem
        for split, filename in FILES.items()
        for problem in load_file(root / filename, split)
    ]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("PHYSICS split IDs must be disjoint")
    return problems


def load_split(split: Split = "test", revision: str = REVISION) -> list[Problem]:
    from datasets import load_dataset

    records = load_dataset(DATASET, split=split, revision=revision)
    return load_problems(records, split)


def validate_source(path: Path) -> None:
    load_directory(path)
