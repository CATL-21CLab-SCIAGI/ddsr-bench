from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, get_args

from ddsr_bench.benchmarks.collect import group_results
from ddsr_bench.benchmarks.hle.data.schemas import AnswerType, Subset
from ddsr_bench.benchmarks.hle.evaluation.metrics import score_results
from ddsr_bench.benchmarks.utils import benchmark_config

_BIN_SIZE = int(benchmark_config("hle")["calibration_bin_size"])


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    """Return official HLE judge fields and the generated response path."""
    fields = result.get("static_result") or {}
    if not fields:
        fields = json.loads(
            (trial / "validation" / "result.json").read_text(encoding="utf-8")
        )
    required = (
        "subset",
        "category",
        "answer_type",
        "correct",
        "confidence",
        "judge_model",
    )
    if any(field not in fields for field in required):
        raise ValueError("HLE trial has incomplete judge fields")
    confidence = fields["confidence"]
    judged = fields.get("judged", "judgment" in fields)
    if fields["subset"] not in get_args(Subset):
        raise ValueError("HLE trial has an invalid verified class")
    if (
        not isinstance(fields["category"], str)
        or not fields["category"]
        or fields["answer_type"] not in get_args(AnswerType)
        or type(fields["correct"]) is not bool
        or isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 100
        or type(judged) is not bool
        or not isinstance(fields["judge_model"], str)
        or not fields["judge_model"]
    ):
        raise ValueError("HLE trial has invalid benchmark fields")
    benchmark = {field: fields[field] for field in required}
    benchmark["judged"] = judged
    return benchmark, trial / "artifacts" / "answer.txt"


def _metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    records = [
        trial["benchmark_config"] | {"status": trial["status"]} for trial in trials
    ]
    return score_results(records, beta=_BIN_SIZE)


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate official HLE accuracy, interval, and calibration metrics."""
    if not trials:
        return {}
    return {
        "overall": _metrics(trials),
        "by_subset": group_results(trials, "subset", _metrics),
        "by_category": group_results(trials, "category", _metrics),
        "by_answer_type": group_results(trials, "answer_type", _metrics),
        "by_attempt": group_results(trials, "attempt", _metrics),
    }
