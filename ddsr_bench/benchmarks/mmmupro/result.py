from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.collect import group_results

_HEADLINE = ("standard10", "vision")


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    fields = result.get("static_result") or {}
    if not fields:
        fields = json.loads(
            (trial / "validation" / "result.json").read_text(encoding="utf-8")
        )
    required = ("setting", "subject", "domain", "selected", "method")
    if any(field not in fields for field in required):
        raise ValueError("MMMU-Pro trial has incomplete benchmark fields")
    return {
        field: fields[field] for field in required
    }, trial / "artifacts" / "answer.txt"


def _metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(trial["status"] == "passed" for trial in trials)
    return {
        "generated": len(trials),
        "passed": passed,
        "accuracy": 100 * passed / len(trials) if trials else None,
        "fallbacks": sum(
            trial["benchmark_config"]["method"] == "seeded-fallback" for trial in trials
        ),
        "errors": sum(trial["status"] == "error" for trial in trials),
    }


def _suite(trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_setting = group_results(trials, "setting", _metrics)
    accuracies = {
        setting: by_setting[setting]["accuracy"]
        for setting in _HEADLINE
        if setting in by_setting
    }
    headline = (
        sum(accuracies.values()) / len(_HEADLINE)
        if len(accuracies) == len(_HEADLINE)
        else None
    )
    return {
        "overall": _metrics(trials),
        "headline_accuracy": headline,
        "by_setting": by_setting,
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {}
    return _suite(trials) | {
        "by_attempt": group_results(trials, "attempt", _suite),
        "by_domain": group_results(trials, "domain", _metrics),
        "by_subject": group_results(trials, "subject", _metrics),
    }
