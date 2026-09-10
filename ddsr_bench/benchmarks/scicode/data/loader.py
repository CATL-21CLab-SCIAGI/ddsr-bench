from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import h5py
import scipy.sparse

from ddsr_bench.benchmarks.scicode.data.schemas import SciCodeProblem, SciCodeStep

_FIXED_STEPS = {"13.6", "62.1", "76.3"}


def _text(record: Mapping[str, Any], key: str, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{label} requires text at {key!r}")
    return value


def _step(record: Mapping[str, Any], problem_id: str) -> SciCodeStep:
    step_id = _text(record, "step_number", problem_id)
    tests = record.get("test_cases")
    if not isinstance(tests, list) or any(not isinstance(test, str) for test in tests):
        raise TypeError(f"SciCode step {step_id!r} requires text test cases")
    return SciCodeStep(
        id=step_id,
        statement=_text(record, "step_description_prompt", step_id),
        function=_text(record, "function_header", step_id),
        return_line=_text(record, "return_line", step_id),
        background=_text(record, "step_background", step_id),
        tests=tuple(tests),
    )


def load_problem(record: Mapping[str, Any]) -> SciCodeProblem:
    """Normalize one problem from the official SciCode dataset."""
    problem_id = _text(record, "problem_id", "SciCode problem")
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
        dependencies=_text(record, "required_dependencies", problem_id),
        background=_text(record, "problem_background_main", problem_id),
        steps=steps,
    )


def load_split(split: str = "test") -> list[SciCodeProblem]:
    """Load one official SciCode split from Hugging Face."""
    from datasets import load_dataset

    records: Iterable[Mapping[str, Any]] = load_dataset("SciCode1/SciCode", split=split)
    return [load_problem(record) for record in records]


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
