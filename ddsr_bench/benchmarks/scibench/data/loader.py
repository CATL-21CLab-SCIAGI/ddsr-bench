from __future__ import annotations

import json
from base64 import b64encode
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, get_args

from ddsr_bench.benchmarks.utils import benchmark_config, read_text

from .schemas import AnswerSpec, Example, Problem, ProblemSpec, Split

_CONFIG = benchmark_config("scibench")
TEXTBOOKS = tuple(map(str, _CONFIG["textbooks"]))
EXPECTED = {str(key): int(value) for key, value in _CONFIG["expected_records"].items()}
EXPECTED_EXAMPLES = {
    str(key): int(value) for key, value in _CONFIG["expected_examples"].items()
}


def _records(path: Path) -> list[Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise TypeError(f"{path} must contain a JSON list of objects")
    return value


def _object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _example(record: Mapping[str, Any], label: str) -> Example:
    return Example(
        read_text(record, "problem_text", label),
        read_text(record, "unit", label),
        read_text(record, "answer_number", label),
        read_text(record, "solution", label),
    )


def _image(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    else:
        raise ValueError(f"{path} has an unsupported image format")
    return f"data:{mime};base64,{b64encode(raw).decode()}"


def load_problem(
    record: Mapping[str, Any],
    *,
    split: Split,
    textbook: str,
    index: int,
    image_path: Path | None = None,
    examples: Sequence[Example] = (),
) -> Problem:
    if split not in get_args(Split):
        raise ValueError(f"unknown SciBench split {split!r}")
    label = f"{split}/{textbook}/{index}"
    problem_id = read_text(record, "problemid", label).strip()
    spec = ProblemSpec(
        id=f"{split}/{textbook}/{index:03d}",
        split=split,
        textbook=textbook,
        problem_id=problem_id,
        question=read_text(record, "problem_text", label),
        unit=read_text(record, "unit", label),
        image=_image(image_path) if image_path is not None else None,
        examples=tuple(examples[:3]),
    )
    answer = AnswerSpec(
        read_text(record, "answer_number", label),
        read_text(record, "unit", label),
    )
    return Problem(spec, answer)


def _text(root: Path) -> tuple[list[Problem], int]:
    problems = []
    example_count = 0
    for textbook in TEXTBOOKS:
        examples = tuple(
            _example(record, f"{textbook}_sol")
            for record in _records(root / "original" / f"{textbook}_sol.json")
        )
        example_count += len(examples)
        problems.extend(
            load_problem(
                record,
                split="text",
                textbook=textbook,
                index=index,
                examples=examples,
            )
            for index, record in enumerate(
                _records(root / "original" / f"{textbook}.json")
            )
        )
    return problems, example_count


def _visual(root: Path) -> tuple[list[Problem], int]:
    examples = list((root / "img" / "problem_solution").glob("*/data.json"))
    problems = []
    directories = sorted(
        (root / "img" / "problem").iterdir(), key=lambda path: int(path.name)
    )
    for directory in directories:
        record = _object(directory / "data.json")
        textbook = read_text(record, "source", directory.name)
        problems.append(
            load_problem(
                record,
                split="image",
                textbook=textbook,
                index=int(directory.name),
                image_path=directory / "img_question.png",
            )
        )
    return problems, len(examples)


def load_directory(
    path: str | Path, splits: Sequence[str] = ("text", "image")
) -> list[Problem]:
    if (
        not splits
        or len(splits) != len(set(splits))
        or any(split not in get_args(Split) for split in splits)
    ):
        raise ValueError("SciBench splits must be unique text/image names")
    root = Path(path) / "dataset"
    loaded = {"text": _text(root), "image": _visual(root)}
    for split, (problems, examples) in loaded.items():
        if len(problems) != EXPECTED[split] or examples != EXPECTED_EXAMPLES[split]:
            raise ValueError(f"SciBench {split} does not match its pinned size")
    problems = [problem for split in splits for problem in loaded[split][0]]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("SciBench normalized IDs must be unique")
    counts = Counter(problem.spec.split for problem in problems)
    if counts != Counter({split: EXPECTED[split] for split in splits}):
        raise ValueError("SciBench source does not match selected splits")
    return problems


def validate_source(path: Path) -> None:
    load_directory(path)
