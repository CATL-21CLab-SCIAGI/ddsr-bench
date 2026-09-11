import json
from pathlib import Path
from typing import Any


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


def _groups(
    trials: list[dict[str, Any]], field: str
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        value = trial[field] if field == "attempt" else trial["benchmark_config"][field]
        groups.setdefault(str(value), []).append(trial)
    return {name: _metrics(groups[name]) for name in sorted(groups)}


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the metrics reported by the official CMPhysBench evaluator."""
    scored = [trial for trial in trials if trial["reward"] is not None]
    if not scored:
        return {}
    return {
        "overall": _metrics(scored),
        "by_topic": _groups(scored, "topic"),
        "by_answer_type": _groups(scored, "answer_type"),
        "by_attempt": _groups(scored, "attempt"),
    }
