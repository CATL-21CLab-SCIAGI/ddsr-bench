from __future__ import annotations

import ast
import string
from base64 import b64encode
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, get_args

from ddsr_bench.benchmarks.utils import benchmark_config, read_text

from .schemas import AnswerSpec, Problem, ProblemSpec, Setting

_CONFIG = benchmark_config("mmmupro")
FILES = {
    str(setting): tuple(map(str, filenames))
    for setting, filenames in _CONFIG["files"].items()
}
EXPECTED = {str(key): int(value) for key, value in _CONFIG["expected_records"].items()}
OFFICIAL_SETTINGS = tuple(map(str, _CONFIG["official_settings"]))
_DOMAINS = {
    "Art and Design": {"Art", "Art_Theory", "Design", "Music"},
    "Business": {"Accounting", "Economics", "Finance", "Manage", "Marketing"},
    "Science": {"Biology", "Chemistry", "Geography", "Math", "Physics"},
    "Health and Medicine": {
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    },
    "Humanities and Social Science": {
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    },
    "Tech and Engineering": {
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    },
}


def _domain(subject: str) -> str:
    matches = [name for name, subjects in _DOMAINS.items() if subject in subjects]
    if len(matches) != 1:
        raise ValueError(f"MMMU-Pro subject {subject!r} has no unique domain")
    return matches[0]


def _image(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not isinstance(value.get("bytes"), bytes):
        raise TypeError(f"{label} requires embedded image bytes")
    raw = value["bytes"]
    if raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    else:
        raise ValueError(f"{label} has an unsupported image format")
    return f"data:{mime};base64,{b64encode(raw).decode()}"


def _choices(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} requires a sequence of choices")
    choices = tuple(value)
    if not 2 <= len(choices) <= len(string.ascii_uppercase) or any(
        not isinstance(item, str) for item in choices
    ):
        raise ValueError(f"{label} has invalid choices")
    return choices


def load_problem(record: Mapping[str, Any], setting: str) -> Problem:
    if setting not in get_args(Setting):
        raise ValueError(f"unknown MMMU-Pro setting {setting!r}")
    raw_id = read_text(record, "id", setting)
    subject = read_text(record, "subject", raw_id)
    choices = _choices(record.get("options"), raw_id)
    if setting == "vision":
        image = _image(record.get("image"), raw_id)
        images = () if image is None else (image,)
        question = ""
        explanation = ""
    else:
        images = tuple(
            image
            for index in range(1, 8)
            if (image := _image(record.get(f"image_{index}"), raw_id)) is not None
        )
        question = read_text(record, "question", raw_id)
        explanation = read_text(record, "explanation", raw_id)
    if not images:
        raise ValueError(f"{raw_id} has no image for {setting}")
    answer = read_text(record, "answer", raw_id)
    if answer not in string.ascii_uppercase[: len(choices)]:
        raise ValueError(f"{raw_id} has an invalid answer")
    spec = ProblemSpec(
        id=f"{setting}/{raw_id}",
        raw_id=raw_id,
        setting=setting,
        subject=subject,
        domain=_domain(subject),
        question=question,
        choices=choices,
        images=images,
    )
    return Problem(spec, AnswerSpec(answer, explanation))


def _parquet(path: Path) -> Iterator[Mapping[str, Any]]:
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
        yield from batch.to_pylist()


def iter_setting(path: str | Path, setting: str) -> Iterator[Problem]:
    """Stream one setting while validating its pinned size and identities."""
    if setting not in FILES:
        raise ValueError(f"unknown MMMU-Pro setting {setting!r}")
    root = Path(path)
    seen = set()
    count = 0
    for filename in FILES[setting]:
        for record in _parquet(root / filename):
            problem = load_problem(record, setting)
            if problem.spec.id in seen:
                raise ValueError(f"MMMU-Pro {setting} has duplicate IDs")
            seen.add(problem.spec.id)
            count += 1
            yield problem
    if count != EXPECTED[setting]:
        raise ValueError(f"MMMU-Pro {setting} does not match its pinned size")


def load_setting(path: str | Path, setting: str) -> list[Problem]:
    return list(iter_setting(path, setting))


def iter_directory(
    path: str | Path, settings: Sequence[str] = OFFICIAL_SETTINGS
) -> Iterator[Problem]:
    if not settings or len(settings) != len(set(settings)):
        raise ValueError("MMMU-Pro settings must be nonempty and unique")
    for setting in settings:
        yield from iter_setting(path, setting)


def load_directory(
    path: str | Path, settings: Sequence[str] = OFFICIAL_SETTINGS
) -> list[Problem]:
    problems = list(iter_directory(path, settings))
    counts = Counter(problem.spec.setting for problem in problems)
    if counts != Counter({setting: EXPECTED[setting] for setting in settings}):
        raise ValueError("MMMU-Pro source does not match selected settings")
    return problems


def validate_source(path: Path) -> None:
    for setting in FILES:
        for _ in iter_setting(path, setting):
            pass
