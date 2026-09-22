"""Dispatch CritPt submissions; normalize attempt selection once here."""

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import ListConfig

from .official import build_batch, submit_batch


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
    backend = config.get("backend", "official")
    if backend not in ("official", "internal"):
        raise ValueError(f"unknown submission backend: {backend!r}")
    job = Path(job_dir)
    suffix = "" if backend == "official" else "-internal"
    label = "-".join(map(str, attempts))
    output = job / f"submission{suffix}-{label}.json"
    if output.exists():
        raise ValueError(f"submission result already exists: {output}")
    logging.getLogger("ddsr_bench").info(
        "Submitting attempts %s to %s", attempts, backend
    )
    if backend == "official":
        result = submit_batch(
            build_batch(job, attempts),
            os.environ.get(str(config["api_key_env"]), ""),
            endpoint=str(config["endpoint"]),
            timeout=float(config["timeout_sec"]),
        )
    else:
        from .internal import submit as submit_internal

        result = submit_internal(job, attempts, **dict(config["internal"]))
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return f"submitted attempts {attempts}; result saved to {output}"
