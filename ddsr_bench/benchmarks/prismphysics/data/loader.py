from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.utils import benchmark_config, read_text

from .schemas import AnswerSpec, FormulaNode, Image, Problem, ProblemSpec, Subquestion

# Expose concise loader names without duplicating the packaged source settings.
_CONFIG = benchmark_config("prismphysics")
DATASET = str(_CONFIG["dataset"])
REVISION = str(_CONFIG["revision"])
FILES = tuple(str(file) for file in _CONFIG["files"])
THRESHOLD = float(_CONFIG["threshold"])
EXPECTED_RECORDS = int(_CONFIG["expected_records"])
DOMAINS = {
    "01": "mechanics",
    "02": "electromagnetism",
    "03": "optics",
    "04": "atomic-nuclear-particle",
    "05": "thermodynamics-statistical",
    "06": "quantum-mechanics",
    "07": "solid-state-miscellaneous",
}


def _items(value: Any, field: str, problem_id: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise TypeError(f"{problem_id} requires a list of objects at {field!r}")
    return value


def _public_text(record: Mapping[str, Any], field: str, problem_id: str) -> str:
    value = record.get(field, "")
    if not isinstance(value, str):
        raise TypeError(f"{problem_id} requires text at {field!r}")
    return value


def _validate_dag(formulas: tuple[FormulaNode, ...], problem_id: str) -> None:
    dependencies = {node.index: node.dependencies for node in formulas}
    if len(dependencies) != len(formulas) or any(
        dependency not in dependencies
        for node in formulas
        for dependency in node.dependencies
    ):
        raise ValueError(f"{problem_id} has an invalid formula DAG")
    visiting = set()
    visited = set()

    def visit(index: int) -> None:
        if index in visiting:
            raise ValueError(f"{problem_id} formula DAG contains a cycle")
        if index in visited:
            return
        visiting.add(index)
        for dependency in dependencies[index]:
            visit(dependency)
        visiting.remove(index)
        visited.add(index)

    for index in dependencies:
        visit(index)


def load_problem(record: Mapping[str, Any], shard: str) -> Problem:
    """Normalize one official record while separating public and verifier data."""
    if shard not in DOMAINS:
        raise ValueError(f"unknown PRISM-Physics shard {shard!r}")
    raw_id = record.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        raise TypeError("PRISM-Physics problem requires an integer ID")
    problem_id = f"{shard}-{raw_id}"
    context = read_text(record, "context", problem_id)
    if not context.strip():
        raise ValueError(f"{problem_id} requires nonempty context")

    subquestions = tuple(
        Subquestion(
            letter=_public_text(item, "letter", problem_id),
            prompt=_public_text(item, "subproblem", problem_id),
        )
        for item in _items(record.get("subquestions"), "subquestions", problem_id)
    )
    images = tuple(
        Image(
            caption=read_text(item, "caption", problem_id),
            location=read_text(item, "location", problem_id),
        )
        for item in _items(record.get("images"), "images", problem_id)
    )
    formulas = tuple(
        FormulaNode(
            index=item["index"],
            formula=read_text(item, "formula", problem_id),
            dependencies=tuple(item["dependency"]),
            final=item.get("is_final_answer") is True,
        )
        for item in _items(
            record.get("grading_standard"), "grading_standard", problem_id
        )
    )
    if any(type(node.index) is not int for node in formulas) or any(
        type(dependency) is not int
        for node in formulas
        for dependency in node.dependencies
    ):
        raise TypeError(f"{problem_id} formula indices must be integers")
    _validate_dag(formulas, problem_id)

    spec = ProblemSpec(
        id=problem_id,
        raw_id=raw_id,
        shard=shard,
        domain=DOMAINS[shard],
        context=context,
        source=read_text(record, "source", problem_id),
        subquestions=subquestions,
        images=images,
    )
    return Problem(spec, AnswerSpec(formulas))


def load_problems(records: Iterable[Mapping[str, Any]], shard: str) -> list[Problem]:
    problems = [load_problem(record, shard) for record in records]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError(f"PRISM-Physics shard {shard} has duplicate IDs")
    return problems


def load_shard(path: str | Path) -> list[Problem]:
    source = Path(path)
    match = re.fullmatch(r"(\d{2})_cleaned_dag\.json", source.name)
    if match is None:
        raise ValueError(f"unexpected PRISM-Physics shard name {source.name!r}")
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"{source} must contain a JSON list")
    return load_problems(records, match.group(1))


def load_split(revision: str = REVISION) -> list[Problem]:
    """Load all pinned official JSON shards from Hugging Face."""
    from huggingface_hub import hf_hub_download

    paths = [
        hf_hub_download(DATASET, filename, repo_type="dataset", revision=revision)
        for filename in FILES
    ]
    return _checked([problem for path in paths for problem in load_shard(path)])


def load_directory(path: str | Path) -> list[Problem]:
    """Load the pinned PRISM-Physics edition from a local shard directory."""
    root = Path(path)
    problems = [
        problem for filename in FILES for problem in load_shard(root / filename)
    ]
    return _checked(problems)


def validate_source(path: Path) -> None:
    """Validate one materialized PRISM-Physics edition."""
    load_directory(path)


def _checked(problems: list[Problem]) -> list[Problem]:
    if len(problems) != EXPECTED_RECORDS:
        raise ValueError("PRISM-Physics does not match its pinned release count")
    return problems
