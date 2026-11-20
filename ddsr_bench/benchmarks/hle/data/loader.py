from __future__ import annotations

import json
from base64 import b64decode
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, get_args

from ddsr_bench.benchmarks.utils import benchmark_config, read_text

from .schemas import AnswerSpec, AnswerType, Problem, ProblemSpec, Subset

_CONFIG = benchmark_config("hle")
DATASET = str(_CONFIG["dataset"])
REVISION = str(_CONFIG["revision"])
FILES = tuple(map(str, _CONFIG["files"]))
SUBSETS = tuple(map(str, _CONFIG["subsets"]))
EXPECTED = {str(key): int(value) for key, value in _CONFIG["expected_records"].items()}
EXPECTED_TYPES = {
    str(key): int(value) for key, value in _CONFIG["expected_answer_types"].items()
}
EXPECTED_IMAGES = int(_CONFIG["expected_images"])
EXPECTED_RATIONALE_IMAGES = int(_CONFIG["expected_rationale_images"])
EXPECTED_SELECTED = int(_CONFIG["expected_selected_records"])
EXPECTED_SELECTED_TYPES = {
    str(key): int(value)
    for key, value in _CONFIG["expected_selected_answer_types"].items()
}
EXPECTED_SELECTED_IMAGES = int(_CONFIG["expected_selected_images"])
EXPECTED_SELECTED_RATIONALE_IMAGES = int(_CONFIG["expected_selected_rationale_images"])


def _image(value: Any, field: str, problem_id: str) -> str | None:
    """Normalize released image encodings to one data-URI representation."""
    if not value:
        return None
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return value
        raise ValueError(f"{problem_id} {field} must be a data URI")
    if not isinstance(value, Mapping) or value.get("path") is not None:
        raise ValueError(f"{problem_id} has an unsupported {field}")
    encoded = value.get("bytes")
    if isinstance(encoded, Mapping):
        encoded = encoded.get("__base64__")
    if not isinstance(encoded, str):
        raise TypeError(f"{problem_id} has an unsupported {field}")
    try:
        header = b64decode(encoded[:64], validate=True)
    except ValueError as error:
        raise ValueError(f"{problem_id} has invalid base64 at {field}") from error
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif header.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif header.startswith((b"II*\x00", b"MM\x00*")):
        mime = "image/tiff"
    elif header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError(f"{problem_id} has an unknown image type at {field}")
    return f"data:{mime};base64,{encoded}"


def _payload(record: Mapping[str, Any], problem_id: str) -> Mapping[str, Any]:
    raw = read_text(record, "json", problem_id)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{problem_id} has invalid source JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{problem_id} source JSON must be an object")
    return value


def load_problem(record: Mapping[str, Any]) -> Problem:
    """Normalize one HLE-Verified row and keep references verifier-side."""
    problem_id = read_text(record, "id", "HLE problem")
    payload = _payload(record, problem_id)
    raw_subset = read_text(record, "Verified_Classes", problem_id)
    subset = raw_subset.removesuffix(" subset")
    answer_type = read_text(payload, "answer_type", problem_id)
    if subset not in get_args(Subset):
        raise ValueError(f"{problem_id} has an unsupported verified class")
    if answer_type not in get_args(AnswerType):
        raise ValueError(f"{problem_id} has an unsupported answer type")

    shared = ("id", "question", "category", "raw_subject")
    if any(record.get(key) != payload.get(key) for key in shared):
        raise ValueError(f"{problem_id} flattened fields disagree with source JSON")
    if str(record.get("answer")) != str(payload.get("answer")):
        raise ValueError(f"{problem_id} flattened answer disagrees with source JSON")
    image = _image(payload.get("image"), "image", problem_id)

    spec = ProblemSpec(
        id=problem_id,
        subset=subset,
        category=read_text(record, "category", problem_id),
        subject=read_text(record, "raw_subject", problem_id),
        question=read_text(record, "question", problem_id),
        image=image,
    )
    answer = AnswerSpec(
        answer=read_text(record, "answer", problem_id),
        type=answer_type,
        rationale=read_text(payload, "rationale", problem_id),
        rationale_image=_image(
            payload.get("rationale_image"), "rationale_image", problem_id
        ),
    )
    return Problem(spec, answer)


def load_problems(records: Iterable[Mapping[str, Any]]) -> list[Problem]:
    problems = [load_problem(record) for record in records]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("HLE problem IDs must be unique")
    counts = Counter(problem.spec.subset for problem in problems)
    types = Counter(problem.answer.type for problem in problems if problem.answer)
    if counts != EXPECTED or types != EXPECTED_TYPES:
        raise ValueError("HLE source does not match its pinned record counts")
    if sum(problem.spec.image is not None for problem in problems) != EXPECTED_IMAGES:
        raise ValueError("HLE source does not match its pinned image count")
    if (
        sum(
            problem.answer is not None and problem.answer.rationale_image is not None
            for problem in problems
        )
        != EXPECTED_RATIONALE_IMAGES
    ):
        raise ValueError("HLE source does not match its pinned rationale-image count")
    return problems


def _parquet(paths: Sequence[str | Path]) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    return load_dataset(
        "parquet", data_files=[str(path) for path in paths], split="train"
    )


def load_directory(path: str | Path) -> list[Problem]:
    root = Path(path)
    return load_problems(_parquet([root / filename for filename in FILES]))


def select(
    problems: Iterable[Problem], subsets: Sequence[str] = SUBSETS
) -> list[Problem]:
    """Select configured verified classes after validating their names."""
    unknown = set(subsets) - set(get_args(Subset))
    if unknown:
        raise ValueError(f"unknown HLE subsets: {sorted(unknown)}")
    selected = [problem for problem in problems if problem.spec.subset in subsets]
    if tuple(subsets) == SUBSETS:
        types = Counter(problem.answer.type for problem in selected if problem.answer)
        if len(selected) != EXPECTED_SELECTED or types != EXPECTED_SELECTED_TYPES:
            raise ValueError("HLE selection does not match its pinned record counts")
        if (
            sum(problem.spec.image is not None for problem in selected)
            != EXPECTED_SELECTED_IMAGES
        ):
            raise ValueError("HLE selection does not match its pinned image count")
        if (
            sum(
                problem.answer is not None
                and problem.answer.rationale_image is not None
                for problem in selected
            )
            != EXPECTED_SELECTED_RATIONALE_IMAGES
        ):
            raise ValueError(
                "HLE selection does not match its pinned rationale-image count"
            )
    return selected


def load_split(revision: str = REVISION) -> list[Problem]:
    """Download the pinned Parquet shards and return the certified selection."""
    from huggingface_hub import hf_hub_download

    paths = [
        hf_hub_download(DATASET, filename, repo_type="dataset", revision=revision)
        for filename in FILES
    ]
    return select(load_problems(_parquet(paths)))


def validate_source(path: Path) -> None:
    load_directory(path)
