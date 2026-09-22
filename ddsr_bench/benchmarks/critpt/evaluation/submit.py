"""Select a CritPt backend; the shared dispatcher saves its report."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .official import build_batch, submit_batch


def submit_attempt(
    job: Path,
    attempt: int,
    defaults: Mapping[str, Any],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Submit saved answers to the selected official or local grader."""
    backend = options.get("backend", "official")
    if backend == "local":
        from .local import submit

        return submit(job, attempt, **dict(defaults["local"]))
    if backend != "official":
        raise ValueError(f"unknown submission backend: {backend!r}")
    return submit_batch(
        build_batch(job, attempt),
        os.environ.get(str(defaults["api_key_env"]), ""),
        endpoint=str(defaults["endpoint"]),
        timeout=float(defaults["timeout_sec"]),
    )
