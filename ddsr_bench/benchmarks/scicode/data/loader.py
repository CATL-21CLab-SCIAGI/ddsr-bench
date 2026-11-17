from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import h5py
import scipy.sparse

from ddsr_bench.benchmarks.scicode.data.schemas import SciCodeProblem, SciCodeStep
from ddsr_bench.benchmarks.utils import benchmark_config, read_jsonl, read_text

_CONFIG = benchmark_config("scicode")
DATASET = str(_CONFIG["dataset"])
REVISION = str(_CONFIG["revision"])
VERSION = str(_CONFIG["version"])
DATA_FILE = str(_CONFIG["data_file"])
TARGETS_FILE = str(_CONFIG["targets_file"])
SPLIT = str(_CONFIG["split"])
EXPECTED_PROBLEMS = int(_CONFIG["expected_problems"])
EXPECTED_SCORED_STEPS = int(_CONFIG["expected_scored_steps"])
_FIXED_STEPS = {"13.6", "62.1", "76.3"}


def _step(record: Mapping[str, Any], problem_id: str) -> SciCodeStep:
    step_id = read_text(record, "step_number", problem_id)
    tests = record.get("test_cases")
    if not isinstance(tests, list) or any(not isinstance(test, str) for test in tests):
        raise TypeError(f"SciCode step {step_id!r} requires text test cases")
    return_line = record.get("return_line")
    if return_line is None and step_id in _FIXED_STEPS:
        return_line = ""
    if not isinstance(return_line, str):
        raise TypeError(f"{step_id} requires text at 'return_line'")
    return SciCodeStep(
        id=step_id,
        statement=read_text(record, "step_description_prompt", step_id),
        function=read_text(record, "function_header", step_id),
        return_line=return_line,
        background=read_text(record, "step_background", step_id),
        tests=tuple(tests),
    )


def load_problem(record: Mapping[str, Any]) -> SciCodeProblem:
    """Normalize one problem from the selected SciCode edition."""
    problem_id = read_text(record, "problem_id", "SciCode problem")
    records = record.get("sub_steps")
    if not isinstance(records, list) or any(
        not isinstance(item, Mapping) for item in records
    ):
        raise TypeError(f"SciCode problem {problem_id!r} requires sub-steps")
    steps = tuple(_step(item, problem_id) for item in records)
    if not steps:
        raise ValueError(f"SciCode problem {problem_id!r} has no sub-steps")
    ids = [step.id for step in steps]
    if len(ids) != len(set(ids)):
        raise ValueError(f"SciCode problem {problem_id!r} has duplicate step IDs")
    return SciCodeProblem(
        id=problem_id,
        dependencies=read_text(record, "required_dependencies", problem_id),
        background=read_text(record, "problem_background_main", problem_id),
        steps=steps,
    )


def load_problems(records: Iterable[Mapping[str, Any]]) -> list[SciCodeProblem]:
    """Normalize SciCode records and require unique problem IDs."""
    problems = [load_problem(record) for record in records]
    ids = [problem.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("SciCode problem IDs must be unique")
    return problems


def load_split(split: str = SPLIT) -> list[SciCodeProblem]:
    """Load the pinned corrected SciCode evaluation split."""
    from huggingface_hub import hf_hub_download

    if split != SPLIT:
        raise ValueError(f"SciCode-Verified provides only the {SPLIT!r} split")
    path = Path(
        hf_hub_download(DATASET, DATA_FILE, repo_type="dataset", revision=REVISION)
    )
    return load_file(path)


def load_file(path: str | Path) -> list[SciCodeProblem]:
    """Load the pinned SciCode edition from a local JSONL file."""
    problems = load_problems(read_jsonl(path))
    scored = sum(
        step.id not in _FIXED_STEPS for problem in problems for step in problem.steps
    )
    if len(problems) != EXPECTED_PROBLEMS or scored != EXPECTED_SCORED_STEPS:
        raise ValueError("SciCode-Verified does not match its pinned release counts")
    return problems


def validate_source(path: Path) -> None:
    """Validate one materialized SciCode edition."""
    problems = load_file(path / DATA_FILE)
    targets = path / TARGETS_FILE
    try:
        with h5py.File(targets, "r") as file:
            for problem in problems:
                for step in problem.steps:
                    if step.id in _FIXED_STEPS:
                        continue
                    for index in range(1, len(step.tests) + 1):
                        if f"{step.id}/test{index}" not in file:
                            raise ValueError(
                                f"SciCode targets require {step.id!r} test {index}"
                            )
    except OSError as error:
        raise ValueError(f"cannot read SciCode targets {targets}") from error


def load_fixed(step_id: str) -> str | None:
    """Load code that SciCode substitutes for its three unavailable steps."""
    if step_id not in _FIXED_STEPS:
        return None
    return (
        files("ddsr_bench.benchmarks.scicode.data")
        .joinpath("fixed", f"{step_id}.txt")
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )


def _sparse(group: h5py.Group) -> scipy.sparse.spmatrix:
    data = group["data"][()]
    shape = tuple(group["shape"][()])
    if "row" in group and "col" in group:
        return scipy.sparse.coo_matrix(
            (data, (group["row"][()], group["col"][()])), shape=shape
        )
    indices, indptr = group["indices"][()], group["indptr"][()]
    if "blocksize" in group:
        return scipy.sparse.bsr_matrix(
            (data, indices, indptr),
            shape=shape,
            blocksize=tuple(group["blocksize"][()]),
        )
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def _dict(group: h5py.Group) -> dict[Any, Any]:
    result = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            item = _sparse(value["sparse_matrix"])
        else:
            item = value[()]
            if isinstance(item, bytes):
                item = item.decode("utf-8", errors="strict")
        try:
            result[float(key)] = item
        except ValueError:
            result[key] = item
    return result


def _group(group: h5py.Group) -> Any:
    first = next(iter(group))
    if first == "list":
        return [value[()] for value in group[first].values()]
    if first == "sparse_matrix":
        return _sparse(group[first])
    return _dict(group)


def _dataset(dataset: h5py.Dataset) -> Any:
    value = dataset[()]
    return value.decode("utf-8", errors="strict") if isinstance(value, bytes) else value


def load_targets(step_id: str, count: int, path: str | Path) -> list[Any]:
    """Decode one step's targets using SciCode's published HDF5 layout."""
    targets = []
    with h5py.File(path, "r") as file:
        for index in range(1, count + 1):
            test = file[f"{step_id}/test{index}"]
            values = list(test.values())
            if len(values) == 1:
                value = values[0]
                if isinstance(value, h5py.Group):
                    targets.append(_group(value))
                else:
                    targets.append(_dataset(value))
            else:
                targets.append(
                    tuple(
                        (
                            _group(value)
                            if isinstance(value, h5py.Group)
                            else _dataset(value)
                        )
                        for value in values
                    )
                )
    return targets
