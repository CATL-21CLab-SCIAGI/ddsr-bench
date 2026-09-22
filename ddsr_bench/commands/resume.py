"""Shared trial recovery helpers; benchmark runners own their resume policy."""

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from ddsr_bench.benchmarks.utils import write_json


def prepare_job(output: Path, config: dict, problems: list, *, resume: bool) -> None:
    # Fingerprint selected inputs, including references, without storing them.
    encoded = json.dumps([asdict(p) for p in problems], sort_keys=True).encode()
    settings = {k: v for k, v in config.items() if k != "n_concurrent_trials"}
    snapshot = {"config": settings, "inputs_sha256": sha256(encoded).hexdigest()}
    path = output / "resume.json"
    if resume:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != snapshot:
            raise ValueError("resume requires unchanged job settings and inputs")
    else:
        output.mkdir(parents=True)
        write_json(path, snapshot)


def completed(directory: Path, task_name: str, attempt: int) -> dict | None:
    """Reuse terminal results, including failures; archive unfinished trials."""
    path = directory / "result.json"
    try:
        trial = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        trial = None
    if trial is not None:
        if trial.get("task_name") != task_name or trial.get("attempt") != attempt:
            raise ValueError(f"invalid completed trial identity: {directory}")
        result = trial.get("static_result")
        if (
            not isinstance(result, dict)
            or "status" not in result
            or "reward" not in result
        ):
            raise ValueError(f"invalid completed trial result: {directory}")
        return result
    if directory.exists():
        archive = directory.parent / ".interrupted"
        archive.mkdir(exist_ok=True)
        directory.rename(archive / f"{directory.name}-{uuid4().hex}")
    return None
