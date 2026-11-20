from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, get_args

from ddsr_bench.benchmarks.utils import benchmark_config, read_text

from .schemas import AnswerSpec, Problem, ProblemSpec, Subject, Track

_CONFIG = benchmark_config("frontierscience")
DATASET = str(_CONFIG["dataset"])
REVISION = str(_CONFIG["revision"])
FILES = tuple(map(str, _CONFIG["files"]))
EXPECTED = {str(key): int(value) for key, value in _CONFIG["expected_records"].items()}
EXPECTED_SUBJECTS = {
    str(key): int(value) for key, value in _CONFIG["expected_subjects"].items()
}


def load_problem(record: Mapping[str, Any], track: str) -> Problem:
    """Normalize one official row without exposing its reference."""
    if track not in get_args(Track):
        raise ValueError(f"unknown FrontierScience track {track!r}")
    task_group_id = read_text(record, "task_group_id", "FrontierScience problem")
    subject = read_text(record, "subject", task_group_id)
    if subject not in get_args(Subject):
        raise ValueError(f"{task_group_id} has an unsupported subject")
    prompt = read_text(record, "problem", task_group_id)
    reference = read_text(record, "answer", task_group_id)
    content = f"{task_group_id}:{prompt}:{reference}"
    spec = ProblemSpec(
        id=sha256(content.encode()).hexdigest(),
        task_group_id=task_group_id,
        track=track,
        subject=subject,
        prompt=prompt,
    )
    answer = AnswerSpec(reference)
    return Problem(spec, answer)


def load_problems(records: Iterable[Mapping[str, Any]], track: str) -> list[Problem]:
    counts: defaultdict[str, int] = defaultdict(int)
    problems = []
    for record in records:
        problem = load_problem(record, track)
        count = counts[problem.spec.id]
        counts[problem.spec.id] += 1
        identifier = problem.spec.id if count == 0 else f"{problem.spec.id}_{count}"
        problems.append(Problem(replace(problem.spec, id=identifier), problem.answer))
    return problems


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} contains invalid JSON") from error
        if not isinstance(record, Mapping):
            raise TypeError(f"{path}:{line_number} must contain an object")
        records.append(record)
    return records


def load_directory(path: str | Path) -> list[Problem]:
    root = Path(path)
    problems = [
        problem
        for filename in FILES
        for problem in load_problems(
            _jsonl(root / filename), Path(filename).parent.name
        )
    ]
    tracks = Counter(problem.spec.track for problem in problems)
    subjects = Counter(problem.spec.subject for problem in problems)
    if tracks != EXPECTED or subjects != EXPECTED_SUBJECTS:
        raise ValueError("FrontierScience source does not match pinned counts")
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("FrontierScience IDs must be unique across tracks")
    return problems


def load_split(revision: str = REVISION) -> list[Problem]:
    """Download and load the complete pinned gold set."""
    from huggingface_hub import hf_hub_download

    paths = [
        hf_hub_download(DATASET, filename, repo_type="dataset", revision=revision)
        for filename in FILES
    ]
    roots = {Path(path).parents[1] for path in paths}
    if len(roots) != 1:
        raise ValueError("FrontierScience files must share one snapshot")
    return load_directory(roots.pop())


def validate_source(path: Path) -> None:
    load_directory(path)
