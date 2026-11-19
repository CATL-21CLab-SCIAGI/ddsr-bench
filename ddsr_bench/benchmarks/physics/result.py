from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, get_args

from ddsr_bench.benchmarks.collect import group_results
from ddsr_bench.benchmarks.physics.data.schemas import AnswerType, Domain, Split
from ddsr_bench.benchmarks.physics.evaluation.metrics import score_results
from ddsr_bench.benchmarks.utils import benchmark_config

_REPETITIONS = int(benchmark_config("physics")["bootstrap_repetitions"])


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    """Return official PHYSICS judge fields and the generated response path."""
    fields = result.get("static_result") or {}
    if not fields:
        fields = json.loads(
            (trial / "validation" / "result.json").read_text(encoding="utf-8")
        )
    required = (
        "split",
        "domain",
        "answer_types",
        "answer_correct",
        "confidence",
        "judge_model",
    )
    if any(field not in fields for field in required):
        raise ValueError("PHYSICS trial has incomplete judge fields")
    answer_types = fields["answer_types"]
    answer_correct = fields["answer_correct"]
    confidence = fields["confidence"]
    judge_model = fields["judge_model"]
    judged = fields.get("judged", "judgment" in fields)
    if fields["split"] not in get_args(Split) or fields["domain"] not in get_args(
        Domain
    ):
        raise ValueError("PHYSICS trial has invalid benchmark fields")
    if (
        not isinstance(answer_types, list)
        or not answer_types
        or any(answer_type not in get_args(AnswerType) for answer_type in answer_types)
        or not isinstance(answer_correct, list)
        or len(answer_correct) != len(answer_types)
        or any(type(flag) is not bool for flag in answer_correct)
    ):
        raise TypeError("PHYSICS trial has invalid answer verdicts")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 100
        or type(judged) is not bool
        or not isinstance(judge_model, str)
        or not judge_model
    ):
        raise ValueError("PHYSICS trial has invalid judgment metadata")
    benchmark = {field: fields[field] for field in required}
    benchmark["judged"] = judged
    return benchmark, trial / "artifacts" / "answer.txt"


def _metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    records = [
        trial["benchmark_config"] | {"status": trial["status"]} for trial in trials
    ]
    return score_results(records, repetitions=_REPETITIONS)


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate strict, answer-level, partial-credit and calibration metrics."""
    if not trials:
        return {}
    return {
        "overall": _metrics(trials),
        "by_split": group_results(trials, "split", _metrics),
        "by_attempt": group_results(trials, "attempt", _metrics),
    }
