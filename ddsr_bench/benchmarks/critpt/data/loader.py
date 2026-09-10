from __future__ import annotations

import re
from pathlib import Path

from ddsr_bench.benchmarks.critpt.data.schemas import (
    AnswerSpec,
    Challenge,
    Problem,
    ProblemSpec,
)

from .utils import read_json, require_text


def _spec(record: dict, path: Path) -> ProblemSpec:
    problem_type = require_text(record, "problem_type", path)
    if problem_type not in ("main", "sub"):
        raise ValueError(f"{path} has unknown problem type {problem_type!r}")
    index = record.get("problem_index")
    if problem_type == "main" and index is not None:
        raise ValueError(f"{path} main problem cannot have an index")
    if problem_type == "sub" and (
        not isinstance(index, int) or isinstance(index, bool) or index < 0
    ):
        raise ValueError(f"{path} subproblem requires a non-negative index")
    return ProblemSpec(
        id=require_text(record, "problem_id", path),
        type=problem_type,
        index=index,
        statement=require_text(record, "problem_description", path),
        code_template=require_text(record, "code_template", path),
        source="critpt",
        source_path=path,
    )


def _problem(record: dict, path: Path) -> Problem:
    snippet = record.get("answer_only_code", "")
    if not isinstance(snippet, str):
        raise TypeError(f"{path} requires text at 'answer_only_code'")
    if not snippet.strip():
        return Problem(spec=_spec(record, path), answer=None)

    testcases = record.get("testcases")
    if testcases is not None and (
        not isinstance(testcases, list)
        or any(not isinstance(testcase, dict) for testcase in testcases)
    ):
        raise TypeError(f"{path} has invalid testcases")
    answer = AnswerSpec(
        code=require_text(record, "answer_code", path),
        snippet=snippet,
        testcases=None if testcases is None else tuple(testcases),
    )
    return Problem(spec=_spec(record, path), answer=answer)


def load_problems(data: dict, source_path: Path) -> tuple[Problem, ...]:
    """Load the problem collection from parsed CritPt challenge data."""
    records = data.get("problems")
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise TypeError(f"{source_path} requires a list at 'problems'")

    problems = tuple(_problem(record, source_path) for record in records)
    mains = tuple(problem for problem in problems if problem.spec.type == "main")
    subs = tuple(problem for problem in problems if problem.spec.type == "sub")
    if len(mains) != 1:
        raise ValueError(f"{source_path} must contain exactly one main problem")
    indices = [problem.spec.index for problem in subs]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{source_path} subproblem indices must be unique")
    return mains + tuple(sorted(subs, key=lambda problem: problem.spec.index))


def load_challenge(path: str | Path) -> Challenge:
    """Load one CritPt challenge and all of its problems."""
    source_path = Path(path)
    data = read_json(source_path)

    return Challenge(
        id=require_text(data, "dataset_name", source_path),
        problems=load_problems(data, source_path),
        source_path=source_path,
    )


def _order(path: Path) -> tuple[bool, int | str]:
    match = re.fullmatch(r"Challenge_(\d+)\.json", path.name)
    return (match is None, path.name if match is None else int(match.group(1)))


def load_challenges(root: str | Path) -> list[Challenge]:
    selected = Path(root)
    paths = (
        [selected]
        if selected.is_file()
        else sorted(selected.rglob("*.json"), key=_order)
    )
    if not paths:
        raise ValueError(f"no challenge JSON files found under {selected}")
    challenges = [load_challenge(path) for path in paths]
    ids = [challenge.id for challenge in challenges]
    if len(ids) != len(set(ids)):
        raise ValueError("challenge IDs must be unique")
    return challenges
