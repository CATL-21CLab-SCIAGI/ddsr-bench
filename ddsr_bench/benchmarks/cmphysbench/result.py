import json
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.collect import group_results


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    """Return CMPhysBench settings and its generated artifact path."""
    fields = result.get("static_result") or {}
    if not fields:
        record = json.loads((trial / "agent" / "response.json").read_text())
        fields = record.get("problem") or {}
    answer_type = fields.get("answer_type")
    topic = fields.get("topic")
    if not all(isinstance(value, str) and value for value in (answer_type, topic)):
        raise ValueError("CMPhysBench trial has incomplete benchmark fields")
    return (
        {"answer_type": answer_type, "topic": topic},
        trial / "artifacts" / "answer.txt",
    )


def _metrics(trials: list[dict[str, Any]]) -> dict[str, float | int]:
    rewards = [float(trial["reward"]) for trial in trials]
    return {
        "trials": len(rewards),
        "mean_seed": 100 * sum(rewards) / len(rewards),
        "accuracy": sum(reward == 1 for reward in rewards) / len(rewards),
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the metrics reported by the official CMPhysBench evaluator."""
    scored = [trial for trial in trials if trial["reward"] is not None]
    if not scored:
        return {}
    return {
        "overall": _metrics(scored),
        "by_topic": group_results(scored, "topic", _metrics),
        "by_answer_type": group_results(scored, "answer_type", _metrics),
        "by_attempt": group_results(scored, "attempt", _metrics),
    }
