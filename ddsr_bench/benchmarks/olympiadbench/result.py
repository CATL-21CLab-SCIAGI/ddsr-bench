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
    required = ("kind", "modality", "subject", "language", "source", "gradable")
    if any(field not in fields for field in required):
        raise ValueError("OlympiadBench trial has incomplete benchmark fields")
    return {
        field: fields[field] for field in required
    }, trial / "artifacts" / "answer.txt"


def _metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [trial for trial in trials if trial["benchmark_config"]["gradable"]]
    passed = sum(trial["status"] == "passed" for trial in graded)
    return {
        "generated": len(trials),
        "graded": len(graded),
        "ungraded": len(trials) - len(graded),
        "passed": passed,
        "accuracy": None if not graded else 100 * passed / len(graded),
        "errors": sum(trial["status"] == "error" for trial in trials),
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {}
    return {
        "overall": _metrics(trials),
        "by_kind": group_results(trials, "kind", _metrics),
        "by_modality": group_results(trials, "modality", _metrics),
        "by_subject": group_results(trials, "subject", _metrics),
        "by_language": group_results(trials, "language", _metrics),
        "by_source": group_results(trials, "source", _metrics),
        "by_attempt": group_results(trials, "attempt", _metrics),
    }
