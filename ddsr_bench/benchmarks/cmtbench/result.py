import json
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.collect import group_results


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    """Return the CMT problem type and generated answer path."""
    fields = result.get("static_result") or {}
    if not fields:
        record = json.loads((trial / "agent" / "response.json").read_text())
        fields = record.get("problem") or {}
    problem_type = fields.get("type")
    if not isinstance(problem_type, str) or not problem_type:
        raise ValueError("CMT-Benchmark trial has no problem type")
    return {"type": problem_type}, trial / "artifacts" / "answer.txt"


def _metrics(trials: list[dict[str, Any]]) -> dict[str, float | int]:
    rewards = [float(trial["reward"]) for trial in trials]
    return {
        "trials": len(rewards),
        "pass_rate": 100 * sum(reward == 1 for reward in rewards) / len(rewards),
        "parse_rate": 100
        * sum(trial["status"] in ("passed", "different") for trial in trials)
        / len(trials),
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the pass rates reported by the official CMT evaluator."""
    scored = [trial for trial in trials if trial["reward"] is not None]
    if not scored:
        return {}
    return {
        "overall": _metrics(scored),
        "by_type": group_results(scored, "type", _metrics),
        "by_attempt": group_results(scored, "attempt", _metrics),
    }
