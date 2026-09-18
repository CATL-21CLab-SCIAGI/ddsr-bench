"""Read flat answers or one native rollout attempt without changing generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .validation import source

PROBLEM = r"Challenge_\d+_main"
TRIAL = re.compile(rf"({PROBLEM})__attempt-(\d+)")
MAX_BYTES = 8_000_000


@dataclass
class Candidate:
    code: str | None
    path: Path
    error: dict | None = None
    generation: dict | None = None
    metadata_errors: list[dict] = field(default_factory=list)


@dataclass
class CandidateBatch:
    answers: dict[str, Candidate]
    layout: str
    attempt: int | None
    available_attempts: list[int]


def _text(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("input file exceeds 8 MB")
    return raw.decode("utf-8")


def _json(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    value = json.loads(_text(path), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value


def _error(stage: str, message: str) -> dict:
    return {"status": "error", "stage": stage, "error": message[:300]}


def _answer(path: Path, problem_id: str) -> Candidate:
    try:
        if path.suffix == ".json":
            data = _json(path)
            if data.get("problem_id", problem_id) != problem_id:
                raise ValueError("candidate ID disagrees with filename")
            text = data.get("generated_code")
        else:
            text = _text(path)
        if not isinstance(text, str):
            raise TypeError("generated_code must be a string")
        if not text.strip():
            raise ValueError("empty candidate answer")
        code = source(text)
        if not code.strip():
            raise ValueError("empty candidate code block")
        return Candidate(code, path)
    except (ValueError, TypeError, OSError, RecursionError) as error:
        return Candidate(
            None, path, _error("input", f"{type(error).__name__}: {error}")
        )


def _metadata(path: Path, candidate: Candidate) -> dict | None:
    if not path.exists():
        return None
    try:
        return _json(path)
    except (ValueError, TypeError, OSError, RecursionError) as error:
        candidate.metadata_errors.append(
            {"source": str(path), "error": str(error)[:300]}
        )
        return None


def _native(directory: Path, problem_id: str, attempt: int) -> Candidate:
    path = directory / "artifacts" / "answer.py"
    candidate = _answer(path, problem_id) if path.exists() else Candidate(None, path)
    trial = _metadata(directory / "result.json", candidate)
    if trial is not None:
        if trial.get("task_name") != f"critpt/{problem_id}" or (
            type(trial.get("attempt")) is not int or trial["attempt"] != attempt
        ):
            candidate.error = _error("input", "trial identity disagrees with directory")
        elif candidate.code is None and candidate.error is None:
            upstream = trial.get("static_result")
            if isinstance(upstream, dict) and upstream.get("status") == "error":
                candidate.error = {
                    **_error(
                        "generation", "rollout did not produce an answer artifact"
                    ),
                    "upstream": upstream,
                }
    if candidate.code is None and candidate.error is None:
        validation = _metadata(directory / "validation" / "result.json", candidate)
        if validation is not None and validation.get("status") == "error":
            candidate.error = {
                **_error("generation", "rollout did not produce an answer artifact"),
                "upstream": validation,
            }
    record = _metadata(directory / "agent" / "response.json", candidate)
    if record is not None:
        if record.get("problem_id") != problem_id:
            candidate.error = _error(
                "input", "response problem ID disagrees with directory"
            )
        responses = record.get("responses")
        if isinstance(responses, list):
            candidate.generation = {
                "model": record.get("model"),
                "strategy": record.get("strategy"),
                "responses": [
                    {
                        "finish_reason": response.get("finish_reason"),
                        "stop_reason": response.get("stop_reason"),
                        "usage": response.get("usage"),
                        "content_empty": not isinstance(response.get("content"), str)
                        or not response["content"].strip(),
                    }
                    for response in responses
                    if isinstance(response, dict)
                ],
            }
    return candidate


def _id(name: str) -> str:
    number = int(name.split("_")[1])
    if not 1 <= number <= 70 or name != f"Challenge_{number}_main":
        raise ValueError(f"unknown or noncanonical problem ID: {name}")
    return name


def load_candidates(directory: Path, attempt: int | None = None) -> CandidateBatch:
    if not directory.is_dir():
        raise ValueError(
            f"candidate directory does not exist or is not a directory: {directory}"
        )
    if attempt is not None and attempt < 0:
        raise ValueError("attempt must be nonnegative")
    paths = [directory, *sorted(directory.rglob("*"))]
    native = [(path, TRIAL.fullmatch(path.name)) for path in paths if path.is_dir()]
    native = [(path, match) for path, match in native if match is not None]
    native_paths = {path for path, _ in native}
    flat = []
    for path in paths:
        if not path.is_file() or path.suffix not in {".py", ".json"}:
            continue
        if any(parent in native_paths for parent in path.parents):
            continue
        name = path.parent.name if path.name == "answer.py" else path.stem
        if re.fullmatch(PROBLEM, name):
            flat.append((_id(name), path))
    if native and flat:
        raise ValueError(
            "cannot combine native rollout trials and flat candidate files"
        )
    available = sorted({int(match[2]) for _, match in native})
    if native:
        if attempt is None:
            if len(available) != 1:
                raise ValueError(
                    f"multiple rollout attempts {available}; select one with --attempt"
                )
            attempt = available[0]
        if attempt not in available:
            raise ValueError(f"attempt {attempt} not found; available: {available}")
        selected = [
            (_id(match[1]), path) for path, match in native if int(match[2]) == attempt
        ]
        if len({path.parent for _, path in selected}) != 1:
            raise ValueError(
                "multiple rollout job directories; select one job directory"
            )
    else:
        if attempt is not None:
            raise ValueError("--attempt requires native rollout trial directories")
        selected = flat
    if not selected:
        raise ValueError("no recognized candidates or rollout trials found")
    answers = {}
    for problem_id, path in selected:
        if problem_id in answers:
            raise ValueError(f"multiple candidate files or trials for {problem_id}")
        answers[problem_id] = (
            _native(path, problem_id, attempt) if native else _answer(path, problem_id)
        )
    return CandidateBatch(
        answers, "native_rollout" if native else "flat", attempt, available
    )
