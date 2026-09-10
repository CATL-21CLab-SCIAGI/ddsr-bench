from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Trial:
    problem_id: str
    agent: str
    model_provider: str
    model: str
    strategy: str
    trial_name: str
    started_at: str
    reward: float | None
    status: str
    answer: str | None
    attempt: int = -1


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _trial(result_path: Path, job: Path) -> Trial:
    data = _read(result_path)
    task_name = data.get("task_name")
    trial_name = data.get("trial_name")
    if not isinstance(task_name, str) or not task_name.startswith("critpt/"):
        raise ValueError(f"{result_path} is not a CritPt trial")
    if not isinstance(trial_name, str) or not trial_name:
        raise ValueError(f"{result_path} has no trial name")

    agent_info = data.get("agent_info") or {}
    model_info = agent_info.get("model_info") or {}
    agent = agent_info.get("name")
    model = model_info.get("name")
    provider = model_info.get("provider") or ""
    agent_config = (data.get("config") or {}).get("agent") or {}
    strategy = (agent_config.get("kwargs") or {}).get("style", "one-step")
    if not all(isinstance(value, str) and value for value in (agent, model, strategy)):
        raise ValueError(f"{result_path} has incomplete agent information")

    static = data.get("static_result")
    rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    reward = static.get("reward") if isinstance(static, dict) else rewards.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, int | float):
        reward = None
    verifier_path = result_path.parent / "verifier" / "result.json"
    verifier = _read(verifier_path) if verifier_path.exists() else {}
    status = (
        static.get("status", "missing")
        if isinstance(static, dict)
        else verifier.get(
            "status", "error" if data.get("exception_info") else "missing"
        )
    )
    attempt = data.get("attempt", -1)
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise TypeError(f"{result_path} has an invalid attempt")
    answer_path = result_path.parent / "artifacts" / "answer.py"
    return Trial(
        problem_id=task_name.removeprefix("critpt/"),
        agent=agent,
        model_provider=str(provider),
        model=model,
        strategy=strategy,
        trial_name=trial_name,
        started_at=str(data.get("started_at") or ""),
        reward=None if reward is None else float(reward),
        status=str(status),
        answer=str(answer_path.relative_to(job)) if answer_path.is_file() else None,
        attempt=attempt,
    )


def collect_trials(job_dir: str | Path) -> dict[str, Any]:
    """Group Harbor or static trials into deterministic attempt batches."""
    job = Path(job_dir)
    trials = [_trial(path, job) for path in job.glob("*/result.json")]
    if not trials:
        raise ValueError(f"no trial results found in {job}")
    identities = {
        (trial.agent, trial.model_provider, trial.model, trial.strategy)
        for trial in trials
    }
    if len(identities) != 1:
        raise ValueError(
            "cannot combine multiple agent, model, or strategy configurations"
        )

    groups: dict[str, list[Trial]] = {}
    for trial in trials:
        groups.setdefault(trial.problem_id, []).append(trial)
    explicit = [trial.attempt >= 0 for trial in trials]
    if any(explicit) and not all(explicit):
        raise ValueError("cannot combine explicit and inferred attempt indices")
    for problem_trials in groups.values():
        problem_trials.sort(key=lambda item: (item.started_at, item.trial_name))

    if all(explicit):
        attempt_sets = [{trial.attempt for trial in items} for items in groups.values()]
        if any(
            len(items) != len(attempts)
            for items, attempts in zip(groups.values(), attempt_sets)
        ):
            raise ValueError("problem attempts must be unique")
        attempts = sorted(set.intersection(*attempt_sets))
    else:
        attempts = list(range(min(map(len, groups.values()))))
    batches = []
    rows = []
    for attempt in attempts:
        batch = []
        for problem_id in sorted(groups):
            if all(explicit):
                trial = next(
                    item for item in groups[problem_id] if item.attempt == attempt
                )
            else:
                trial = replace(groups[problem_id][attempt], attempt=attempt)
            batch.append(trial)
        rows.extend(batch)
        batches.append(
            {
                "attempt": attempt,
                "validated": sum(trial.status == "validated" for trial in batch),
                "passed": sum(trial.reward == 1 for trial in batch),
                "total": len(batch),
                "trials": [asdict(trial) for trial in batch],
            }
        )

    summary = {
        "problems": len(groups),
        "trials": len(trials),
        "complete_batches": len(batches),
        "unbatched_trials": len(trials) - len(rows),
        "batches": batches,
    }
    (job / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (job / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        fields = tuple(Trial.__dataclass_fields__)
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(trial) for trial in rows)
    return summary
