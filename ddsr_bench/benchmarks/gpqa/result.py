from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.collect import group_results


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    fields = result.get("static_result") or {}
    if not fields:
        fields = json.loads(
            (trial / "validation" / "result.json").read_text(encoding="utf-8")
        )
    required = ("cohort", "domain", "subdomain", "selected")
    if any(field not in fields for field in required):
        raise ValueError("GPQA trial has incomplete benchmark fields")
    return {
        field: fields[field] for field in required
    }, trial / "artifacts" / "answer.txt"


def _metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(trial["status"] == "passed" for trial in trials)
    return {
        "generated": len(trials),
        "passed": passed,
        "accuracy": 100 * passed / len(trials) if trials else None,
        "errors": sum(trial["status"] == "error" for trial in trials),
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {}
    return {
        "overall": _metrics(trials),
        "by_attempt": group_results(trials, "attempt", _metrics),
        "by_domain": group_results(trials, "domain", _metrics),
        "by_subdomain": group_results(trials, "subdomain", _metrics),
    }
