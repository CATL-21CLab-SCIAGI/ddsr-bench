"""Dispatch CritPt submissions; normalize attempt selection once here."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import ListConfig

from ddsr_bench.benchmarks.collect import collect_trials, load_batch
from ddsr_bench.benchmarks.utils import write_json
from ddsr_bench.generation.client import read_api_key

from ..data.schemas import CritPtSource
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
    grader = config.get("grader")
    supported = (
        ("official",) if backend == "official" else ("consensus", "ai", "rule", "llm")
    )
    if grader is not None and grader not in supported:
        raise ValueError(f"unsupported grader for {backend}: {grader!r}")
    job = Path(job_dir)
    # Both destinations consume the same collection; never repair it implicitly.
    if not (job / "summary.json").exists():
        collect_trials(job)
    if backend == "internal" and grader is None:
        sources = {
            CritPtSource(trial.get("benchmark_config", {}).get("source", "critpt"))
            for attempt in attempts
            for trial in load_batch(job, attempt)
        }
        if len(sources) != 1:
            raise ValueError("internal submission requires one collected data source")
        grader = "ai" if sources == {CritPtSource.AI} else "consensus"
    ai = grader in ("ai", "rule", "llm")
    suffix = "" if backend == "official" else "-internal"
    label = "-".join(map(str, attempts))
    # Resolved grader identity is identical for default and explicit selection.
    if backend == "internal":
        suffix += f"-{grader}"
    output = job / f"submission{suffix}-{label}.json"
    pending = job / f"submission{suffix}-{label}.pending.json"
    if output.exists():
        raise ValueError(f"submission result already exists: {output}")
    if backend == "official" and pending.exists():
        raise ValueError(f"submission already pending; reconcile with AA: {pending}")
    if backend == "internal" and pending.exists():
        raise ValueError(
            f"internal submission already pending; reconcile first: {pending}"
        )
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
    elif ai:
        from .ai import submit_ai

        result = submit_ai(
            job,
            attempts,
            pending=pending,
            grader=grader if grader in ("rule", "llm") else None,
            datasets=config.get("datasets"),
            **dict(config["ai"]),
        )
    else:
        from .consensus import submit_consensus

        result = submit_consensus(job, attempts, **dict(config["consensus"]))
    if backend == "internal":
        result["grader"] = grader
    write_json(output, result, overwrite=False)
    if backend == "official" or ai:
        pending.unlink()
    return f"submitted attempts {attempts}; result saved to {output}"
