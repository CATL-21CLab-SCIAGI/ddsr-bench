from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, get_args

from ddsr_bench.benchmarks.collect import group_results
from ddsr_bench.benchmarks.frontierscience.data.schemas import Subject, Track
from ddsr_bench.benchmarks.frontierscience.evaluation.metrics import score_results


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    """Return official FrontierScience judge fields and response path."""
    fields = result.get("static_result") or {}
    if not fields:
        fields = json.loads(
            (trial / "validation" / "result.json").read_text(encoding="utf-8")
        )
    required = ("track", "subject", "correct", "points", "judge_model")
    if any(field not in fields for field in required):
        raise ValueError("FrontierScience trial has incomplete judge fields")
    points = fields["points"]
    judged = fields.get("judged", False)
    if (
        fields["track"] not in get_args(Track)
        or fields["subject"] not in get_args(Subject)
        or type(fields["correct"]) is not bool
        or isinstance(points, bool)
        or not isinstance(points, int | float)
        or not math.isfinite(points)
        or not 0 <= points <= 10
        or type(judged) is not bool
        or not isinstance(fields["judge_model"], str)
        or not fields["judge_model"]
    ):
        raise ValueError("FrontierScience trial has invalid benchmark fields")
    benchmark = {field: fields[field] for field in required}
    benchmark["judged"] = judged
    benchmark["reward"] = float(fields.get("reward", 0))
    return benchmark, trial / "artifacts" / "answer.txt"


def _metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    records = [
        trial["benchmark_config"] | {"status": trial["status"]} for trial in trials
    ]
    return score_results(records)


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {}
    return {
        "overall": _metrics(trials),
        "by_track": group_results(trials, "track", _metrics),
        "by_subject": group_results(trials, "subject", _metrics),
        "by_attempt": group_results(trials, "attempt", _metrics),
    }
