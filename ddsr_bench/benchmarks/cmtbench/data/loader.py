from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, get_args

from ddsr_bench.benchmarks.utils import read_text

from .schemas import AnswerSpec, Problem, ProblemSpec, ProblemType

DATASET = "CMTBenchmark/cmt_benchmark"
REVISION = "c6aa803a6ca553091095ecf07e747b8720b1fbc2"
VERSION = "2.0.0"
DATA_FILE = "cmt_data.jsonl"


def load_problem(record: Mapping[str, Any]) -> Problem:
    """Normalize one record from the official CMT-Benchmark dataset."""
    index = record.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("CMT-Benchmark problem requires an integer index")
    if index < 0:
        raise ValueError("CMT-Benchmark problem index must be nonnegative")
    problem_id = str(index)
    problem_type = read_text(record, "type", problem_id)
    if problem_type not in get_args(ProblemType):
        raise ValueError(f"unsupported CMT-Benchmark problem type {problem_type!r}")

    prompt = read_text(record, "prompt", problem_id)
    solution = read_text(record, "solution", problem_id)
    if not prompt.strip() or not solution.strip():
        raise ValueError(f"{problem_id} requires a prompt and solution")

    spec = ProblemSpec(id=problem_id, prompt=prompt, type=problem_type)
    answer = AnswerSpec(
        solution=solution,
        parameters=read_text(record, "parameters", problem_id),
        functions=read_text(record, "functions", problem_id),
    )
    return Problem(spec=spec, answer=answer)


def load_problems(records: Iterable[Mapping[str, Any]]) -> list[Problem]:
    """Normalize CMT-Benchmark records and require unique indices."""
    problems = [load_problem(record) for record in records]
    ids = [problem.spec.id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError("CMT-Benchmark problem IDs must be unique")
    return problems


def load_split(split: str = "train", revision: str = REVISION) -> list[Problem]:
    """Load the pinned official CMT-Benchmark JSONL file."""
    from datasets import load_dataset

    records: Iterable[Mapping[str, Any]] = load_dataset(
        DATASET,
        data_files=DATA_FILE,
        split=split,
        revision=revision,
    )
    return load_problems(records)
