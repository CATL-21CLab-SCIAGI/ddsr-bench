from __future__ import annotations

import re
from base64 import b64encode
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.utils import benchmark_config, read_text

from .schemas import AnswerSpec, Problem, ProblemSpec

_CONFIG = benchmark_config("olympiadbench")
DATASET = str(_CONFIG["dataset"])
REVISION = str(_CONFIG["revision"])
FILES = tuple(map(str, _CONFIG["files"]))
EXPECTED = {str(key): int(value) for key, value in _CONFIG["expected_records"].items()}


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


def _texts(value: Any, field: str, label: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{label} requires a list of text at {field!r}")
    return tuple(value)


def load_problem(record: Mapping[str, Any], subset: str) -> Problem:
    """Normalize one official row into public and verifier projections."""
    parts = subset.split("_")
    if len(parts) != 5:
        raise ValueError(f"invalid OlympiadBench subset {subset!r}")
    kind, modality, subject, language, source = parts
    raw_id = record.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        raise TypeError(f"{subset} problem requires an integer ID")
    problem_id = f"{subset}/{raw_id}"
    context = record.get("context")
    if context is not None and not isinstance(context, str):
        raise TypeError(f"{problem_id} has invalid context")
    images = tuple(
        image
        for index in range(1, 10)
        if (image := _image(record.get(f"image_{index}"), problem_id)) is not None
    )
    image_indices = {
        index for index in range(1, 10) if record.get(f"image_{index}") is not None
    }
    expected = {
        "modality": "Multimodal" if modality == "MM" else "Text-only",
        "subject": "Math" if subject == "maths" else "Physics",
        "language": "English" if language == "en" else "Chinese",
        "question_type": "Open-ended" if kind == "OE" else "Theorem proof",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{problem_id} metadata disagrees with its subset")
    answer_type = record.get("answer_type")
    multiple_answers = record.get("is_multiple_answer")
    unit = record.get("unit")
    if answer_type is not None and not isinstance(answer_type, str):
        raise TypeError(f"{problem_id} has invalid answer type")
    if multiple_answers is not None and type(multiple_answers) is not bool:
        raise TypeError(f"{problem_id} has invalid multiple-answer flag")
    if unit is not None and not isinstance(unit, str):
        raise TypeError(f"{problem_id} has invalid unit")
    spec = ProblemSpec(
        id=problem_id,
        raw_id=raw_id,
        subset=subset,
        kind=kind,
        modality=modality,
        subject=subject,
        language=language,
        source=source,
        subfield=read_text(record, "subfield", problem_id),
        context=context,
        question=read_text(record, "question", problem_id),
        images=images,
        answer_type=answer_type,
        multiple_answers=multiple_answers,
        unit=unit,
    )
    used_images = {
        int(index)
        for index in re.findall(r"<image_(\d+)>", f"{context or ''}\n{spec.question}")
    }
    if image_indices != set(range(1, len(images) + 1)) or used_images != image_indices:
        raise ValueError(f"{problem_id} has inconsistent image placeholders")
    tolerance = record.get("error")
    if tolerance is not None and not isinstance(tolerance, str):
        raise TypeError(f"{problem_id} has invalid tolerance")
    answer = AnswerSpec(
        solutions=_texts(record.get("solution"), "solution", problem_id) or (),
        answers=_texts(record.get("final_answer"), "final_answer", problem_id),
        tolerance=tolerance,
    )
    return Problem(spec, answer)


def load_problems(records: Iterable[Mapping[str, Any]], subset: str) -> list[Problem]:
    problems = [load_problem(record, subset) for record in records]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError(f"OlympiadBench subset {subset!r} has duplicate IDs")
    return problems


def _parquet(path: Path) -> list[Mapping[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def load_directory(path: str | Path) -> list[Problem]:
    root = Path(path)
    problems = []
    counts = Counter()
    for filename in FILES:
        subset = Path(filename).stem
        loaded = load_problems(_parquet(root / filename), subset)
        counts[subset] = len(loaded)
        problems.extend(loaded)
    if counts != EXPECTED:
        raise ValueError("OlympiadBench source does not match pinned subset counts")
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("OlympiadBench IDs must be unique")
    if sum(problem.spec.kind == "OE" for problem in problems) != int(
        _CONFIG["expected_open_ended"]
    ):
        raise ValueError("OlympiadBench source does not match pinned composition")
    if (
        sum(problem.spec.kind == "TP" for problem in problems)
        != int(_CONFIG["expected_proofs"])
        or sum(problem.spec.modality == "MM" for problem in problems)
        != int(_CONFIG["expected_multimodal"])
        or len(problems) != int(_CONFIG["expected_total"])
    ):
        raise ValueError("OlympiadBench source does not match pinned composition")
    return problems


def load_split(revision: str = REVISION) -> list[Problem]:
    """Load the complete pinned official release from Hugging Face."""
    from huggingface_hub import hf_hub_download

    paths = [
        hf_hub_download(DATASET, name, repo_type="dataset", revision=revision)
        for name in FILES
    ]
    roots = {Path(path).parents[2] for path in paths}
    if len(roots) != 1:
        raise ValueError("OlympiadBench files must share one snapshot")
    return load_directory(roots.pop())


def validate_source(path: Path) -> None:
    load_directory(path)
