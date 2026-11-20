from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ddsr_bench.benchmarks.collect import group_results
from ddsr_bench.benchmarks.hipho.data.loader import PUBLISHED_EXAMS


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    fields = result.get("static_result") or {}
    if not fields:
        fields = json.loads(
            (trial / "validation" / "result.json").read_text(encoding="utf-8")
        )
    required = (
        "exam",
        "field",
        "modality",
        "points",
        "max_points",
        "fine_grained_points",
        "coarse_grained_points",
        "judge_model",
    )
    if any(field not in fields for field in required):
        raise ValueError("HiPhO trial has incomplete benchmark fields")
    for field in (
        "points",
        "max_points",
        "fine_grained_points",
        "coarse_grained_points",
    ):
        value = fields[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError("HiPhO trial has invalid score fields")
    return {
        field: fields[field] for field in required
    }, trial / "artifacts" / "answer.txt"


def _metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {}
    attempts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        attempts[int(trial["attempt"])].append(trial)
    leaderboard, rates = [], []
    for rows in attempts.values():
        per_exam: dict[str, float] = defaultdict(float)
        points = maximum = 0.0
        for trial in rows:
            fields = trial["benchmark_config"]
            per_exam[fields["exam"]] += fields["points"]
            points += fields["points"]
            maximum += fields["max_points"]
        leaderboard.append(mean(per_exam.values()))
        rates.append(100 * points / maximum if maximum else 0.0)
    return {
        "generated": len(trials),
        "attempts": len(attempts),
        "leaderboard_points_mean": mean(leaderboard),
        "score_rate": mean(rates),
        "errors": sum(trial["status"] == "error" for trial in trials),
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {}
    published = [
        trial
        for trial in trials
        if trial["benchmark_config"]["exam"] in PUBLISHED_EXAMS
    ]
    return {
        "current": _metrics(trials),
        "published": _metrics(published),
        "by_attempt": group_results(trials, "attempt", _metrics),
        "by_exam": group_results(trials, "exam", _metrics),
        "by_field": group_results(trials, "field", _metrics),
        "by_modality": group_results(trials, "modality", _metrics),
    }
