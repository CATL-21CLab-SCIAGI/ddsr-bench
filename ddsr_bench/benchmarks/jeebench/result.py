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
    required = ("paper", "subject", "response_type", "selected")
    if any(field not in fields for field in required):
        raise ValueError("JEEBench trial has incomplete benchmark fields")
    return {
        field: fields[field] for field in required
    }, trial / "artifacts" / "answer.txt"


def _metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(trial["status"] == "passed" for trial in trials)
    return {
        "generated": len(trials),
        "full_credit": passed,
        "score": (
            100 * sum(trial["reward"] for trial in trials) / len(trials)
            if trials
            else None
        ),
        "errors": sum(trial["status"] == "error" for trial in trials),
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {}
    return {
        "overall": _metrics(trials),
        "by_attempt": group_results(trials, "attempt", _metrics),
        "by_subject": group_results(trials, "subject", _metrics),
        "by_response_type": group_results(trials, "response_type", _metrics),
        "by_paper": group_results(trials, "paper", _metrics),
    }
