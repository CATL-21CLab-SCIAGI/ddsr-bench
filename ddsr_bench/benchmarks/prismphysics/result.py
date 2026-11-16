import json
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.collect import group_results


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    """Return public PRISM metadata and the generated answer path."""
    fields = result.get("static_result") or {}
    if not fields:
        response = json.loads((trial / "agent" / "response.json").read_text())
        fields = response.get("problem") or {}
    domain, shard = fields.get("domain"), fields.get("shard")
    if not isinstance(domain, str) or not isinstance(shard, str):
        raise TypeError("PRISM-Physics trial has no domain or shard")
    return {"domain": domain, "shard": shard}, trial / "artifacts" / "answer.txt"


def _metrics(trials: list[dict[str, Any]]) -> dict[str, float | int]:
    rewards = [float(trial["reward"]) for trial in trials]
    return {
        "trials": len(rewards),
        "step_score": 100 * sum(rewards) / len(rewards),
        "accuracy": 100
        * sum(trial["status"] == "passed" for trial in trials)
        / len(trials),
        "parse_rate": 100
        * sum(trial["status"] in ("passed", "different") for trial in trials)
        / len(trials),
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate PRISM step score and thresholded problem accuracy."""
    scored = [trial for trial in trials if trial["reward"] is not None]
    if not scored:
        return {}
    return {
        "overall": _metrics(scored),
        "by_domain": group_results(scored, "domain", _metrics),
        "by_shard": group_results(scored, "shard", _metrics),
        "by_attempt": group_results(scored, "attempt", _metrics),
    }
