from __future__ import annotations

import json
from base64 import b64encode
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, get_args

from ddsr_bench.benchmarks.utils import benchmark_config, read_text

from .schemas import AnswerSpec, Language, Problem, ProblemSpec

_CONFIG = benchmark_config("hipho")
DATASET = str(_CONFIG["dataset"])
REVISION = str(_CONFIG["revision"])
EXPECTED = {
    str(exam): int(count) for exam, count in _CONFIG["expected_records"].items()
}
EXAMS = tuple(EXPECTED)
PUBLISHED_EXAMS = tuple(map(str, _CONFIG["published_exams"]))
_CHINESE = {"CPhO_2025", "PanMechanics_2024", "PanMechanics_2025"}


def _texts(value: Any, field: str, problem_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{problem_id} requires a list of text at {field!r}")
    return tuple(value)


def _images(value: Any, root: Path, problem_id: str) -> tuple[str, ...]:
    paths = _texts(value, "image_question", problem_id)
    images = []
    for relative in paths:
        path = root / relative
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"{problem_id} references missing image {relative!r}"
            ) from error
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"{problem_id} references a non-PNG image")
        images.append(f"data:image/png;base64,{b64encode(raw).decode()}")
    return tuple(images)


def load_problem(
    record: Mapping[str, Any],
    exam: str,
    information: str,
    root: Path,
) -> Problem:
    """Normalize one official row into public and verifier projections."""
    problem_id = read_text(record, "id", exam)
    if record.get("source") != exam:
        raise ValueError(f"{problem_id} source disagrees with its exam file")
    language: Language = "zh" if exam in _CHINESE else "en"
    if language not in get_args(Language):
        raise AssertionError("unreachable")
    spec = ProblemSpec(
        id=problem_id,
        exam=exam,
        language=language,
        information=information,
        context=read_text(record, "context", problem_id),
        question=read_text(record, "question", problem_id),
        modality=read_text(record, "modality", problem_id),
        field=read_text(record, "field", problem_id),
        images=_images(record.get("image_question"), root, problem_id),
    )
    units = record.get("unit")
    points = record.get("points")
    marking = record.get("marking", [])
    if not isinstance(units, list) or any(
        item is not None and not isinstance(item, str) for item in units
    ):
        raise TypeError(f"{problem_id} has invalid answer units")
    if not isinstance(points, list) or any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in points
    ):
        raise TypeError(f"{problem_id} has invalid point values")
    if not isinstance(marking, list):
        raise TypeError(f"{problem_id} has an invalid marking scheme")
    answer = AnswerSpec(
        answers=_texts(record.get("answer"), "answer", problem_id),
        answer_types=_texts(record.get("answer_type"), "answer_type", problem_id),
        units=tuple(units),
        points=tuple(map(float, points)),
        marking=tuple(marking),
    )
    return Problem(spec, answer)


def load_problems(
    records: Iterable[Mapping[str, Any]], exam: str, root: Path
) -> list[Problem]:
    rows = list(records)
    if not rows or set(rows[0]) != {"information"}:
        raise ValueError(f"HiPhO exam {exam!r} requires one information row")
    information = read_text(rows[0], "information", exam)
    problems = [load_problem(row, exam, information, root) for row in rows[1:]]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError(f"HiPhO exam {exam!r} has duplicate IDs")
    return problems


def _records(path: Path) -> list[Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise TypeError(f"{path} must contain a JSON list of objects")
    return value


def load_directory(path: str | Path) -> list[Problem]:
    root = Path(path) / "data"
    if not root.is_dir():
        raise ValueError(f"{path} does not contain the HiPhO data directory")
    problems = []
    counts = Counter()
    for exam in EXAMS:
        loaded = load_problems(_records(root / f"{exam}.json"), exam, root)
        counts[exam] = len(loaded)
        problems.extend(loaded)
    if counts != EXPECTED or len(problems) != int(_CONFIG["expected_total"]):
        raise ValueError("HiPhO source does not match pinned exam counts")
    if len({path.name for path in (root / "image_question").glob("*.png")}) != int(
        _CONFIG["expected_images"]
    ):
        raise ValueError("HiPhO source does not match its pinned image count")
    if sum(problem.spec.exam in PUBLISHED_EXAMS for problem in problems) != int(
        _CONFIG["expected_published"]
    ):
        raise ValueError("HiPhO source does not match its published cohort")
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("HiPhO problem IDs must be unique")
    return problems


def load_split(revision: str = REVISION) -> list[Problem]:
    """Load the complete pinned official release from Hugging Face."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        DATASET,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["data/*.json", "data/image_question/*.png"],
    )
    return load_directory(root)


def validate_source(path: Path) -> None:
    load_directory(path)
