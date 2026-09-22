import json
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.collect import group_results


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    """Return the PHYBench tag and generated answer path."""
    fields = result.get("static_result") or {}
    if not fields:
        record = json.loads((trial / "agent" / "response.json").read_text())
        fields = record.get("problem") or {}
    tag = fields.get("tag")
    if not isinstance(tag, str) or not tag:
        raise ValueError("PHYBench trial has no problem tag")
    return {"tag": tag}, trial / "artifacts" / "answer.txt"


def _metrics(trials: list[dict[str, Any]]) -> dict[str, float | int]:
    rewards = [float(trial["reward"]) for trial in trials]
    return {
        "trials": len(rewards),
        "mean_eed": 100 * sum(rewards) / len(rewards),
        "accuracy": sum(reward == 1 for reward in rewards) / len(rewards),
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate PHYBench's EED and exact-match metrics."""
    scored = [trial for trial in trials if trial["reward"] is not None]
    if not scored:
        return {}
    return {
        "overall": _metrics(scored),
        "by_tag": group_results(scored, "tag", _metrics),
        "by_attempt": group_results(scored, "attempt", _metrics),
    }
