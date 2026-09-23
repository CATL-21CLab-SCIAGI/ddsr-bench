"""Dispatch CritPt submissions; normalize attempt selection once here."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import ListConfig

from ddsr_bench.benchmarks.collect import collect_trials
from ddsr_bench.benchmarks.utils import write_json
from ddsr_bench.generation.client import read_api_key

from .official import build_batch, submit_official


def submit(job_dir: str | Path, config: Mapping[str, Any]) -> str:
    attempts = config.get("attempts")
    if type(attempts) is int:
        attempts = [attempts]
    if not isinstance(attempts, (list, ListConfig)):
        raise TypeError("benchmark.submission.attempts must be an integer or list")
    attempts = list(attempts)
    if not attempts or any(type(i) is not int or i < 0 for i in attempts):
        raise ValueError("attempts must be nonempty nonnegative integers")
    if len(set(attempts)) != len(attempts):
        raise ValueError("attempts must be unique")
    attempts.sort()  # Selection is set-like; order must not bypass duplicate checks.
    backend = config.get("backend", "official")
    if backend not in ("official", "internal"):
        raise ValueError(f"unknown submission backend: {backend!r}")
    job = Path(job_dir)
    suffix = "" if backend == "official" else "-internal"
    label = "-".join(map(str, attempts))
    output = job / f"submission{suffix}-{label}.json"
    pending = job / f"submission-{label}.pending.json"
    if output.exists():
        raise ValueError(f"submission result already exists: {output}")
    if backend == "official" and pending.exists():
        raise ValueError(f"submission already pending; reconcile with AA: {pending}")
    # Both destinations consume the same collection; never repair it implicitly.
    if not (job / "summary.json").exists():
        collect_trials(job)
    logging.getLogger("ddsr_bench").info(
        "Submitting attempts %s to %s", attempts, backend
    )
    if backend == "official":
        payload = build_batch(job, attempts)
        api_key = read_api_key(config["api_key_env"]) or ""
        if not api_key:
            raise ValueError("an Artificial Analysis API key is required")
        endpoint, timeout = str(config["endpoint"]), float(config["timeout_sec"])
        try:
            write_json(
                pending, {"attempts": attempts, "endpoint": endpoint}, overwrite=False
            )
        except FileExistsError as error:
            raise ValueError(f"submission already pending: {pending}") from error
        # A competing request may have finished between the first check and claim.
        if output.exists():
            pending.unlink()
            raise ValueError(f"submission result already exists: {output}")
        # Keep the claim on request/save failure: AA may already have accepted it.
        result = submit_official(payload, api_key, endpoint=endpoint, timeout=timeout)
    else:
        from .internal import submit_internal

        result = submit_internal(job, attempts, **dict(config["internal"]))
    write_json(output, result, overwrite=False)
    if backend == "official":
        pending.unlink()
    return f"submitted attempts {attempts}; result saved to {output}"
