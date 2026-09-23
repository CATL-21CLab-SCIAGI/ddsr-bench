from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.registry import result_adapter, summarizer
from ddsr_bench.benchmarks.utils import read_json, write_json


def group_results(
    trials: list[dict[str, Any]],
    field: str,
    metrics: Callable[[list[dict[str, Any]]], dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    """Summarize groups using the benchmark's own metric calculation."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        value = trial[field] if field == "attempt" else trial["benchmark_config"][field]
        groups.setdefault(str(value), []).append(trial)
    return {name: metrics(groups[name]) for name in sorted(groups)}


@dataclass(frozen=True, slots=True)
class Trial:
    benchmark: str
    problem_id: str
    agent: str
    model_provider: str
    model: str
    benchmark_config: dict[str, Any]
    trial_name: str
    started_at: str
    reward: float | None
    status: str
    artifact: str | None
    attempt: int = -1
    # Preserve the failing static/verifier result, or Harbor exception details.
    error: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """A trial's raw result, including read failures for tolerant adapters."""

    path: Path
    data: dict[str, Any] | None
    error: Exception | None = None


def _trial(record: TrialRecord, job: Path) -> Trial:
    if record.error is not None:
        raise record.error
    result_path, data = record.path, record.data
    assert data is not None
    task_name = data.get("task_name")
    trial_name = data.get("trial_name")
    if not isinstance(task_name, str):
        raise TypeError(f"{result_path} has no task name")
    benchmark, separator, problem_id = task_name.partition("/")
    if not separator or not problem_id:
        raise ValueError(f"{result_path} has an invalid task name")
    try:
        fields = result_adapter(benchmark)
    except ValueError as error:
        raise ValueError(f"{result_path} has an unsupported benchmark") from error
    if not isinstance(trial_name, str) or not trial_name:
        raise ValueError(f"{result_path} has no trial name")

    agent_info = data.get("agent_info") or {}
    model_info = agent_info.get("model_info") or {}
    agent = agent_info.get("name")
    model = model_info.get("name")
    provider = model_info.get("provider") or ""
    benchmark_config, artifact_path = fields(data, result_path.parent)
    if not all(isinstance(value, str) and value for value in (agent, model)):
        raise ValueError(f"{result_path} has incomplete agent information")

    static = data.get("static_result")
    rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    reward = static.get("reward") if isinstance(static, dict) else rewards.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, int | float):
        reward = None
    verifier_path = result_path.parent / "verifier" / "result.json"
    verifier = read_json(verifier_path) if verifier_path.exists() else {}
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
    failure = static if isinstance(static, dict) else verifier
    return Trial(
        benchmark=benchmark,
        problem_id=problem_id,
        agent=agent,
        model_provider=str(provider),
        model=model,
        benchmark_config=benchmark_config,
        trial_name=trial_name,
        started_at=str(data.get("started_at") or ""),
        reward=None if reward is None else float(reward),
        status=str(status),
        artifact=(
            str(artifact_path.relative_to(job)) if artifact_path.is_file() else None
        ),
        attempt=attempt,
        error=(failure or data.get("exception_info")) if status == "error" else None,
    )


def load_trials[T](
    job_dir: str | Path,
    *,
    adapter: Callable[[TrialRecord, Path], T] = _trial,
    paths: Iterable[Path] | None = None,
    reader: Callable[[Path], dict[str, Any]] = read_json,
) -> list[T]:
    """Read trials without filtering attempts or writing collection summaries.

    Adapters decide whether malformed/missing results are fatal. A caller may
    supply discovered paths (including missing results) and a bounded reader.
    """
    job = Path(job_dir)
    records = []
    for path in job.glob("*/result.json") if paths is None else paths:
        try:
            record = TrialRecord(path, reader(path))
        except (OSError, ValueError, TypeError, RecursionError) as error:
            record = TrialRecord(path, None, error)
        records.append(adapter(record, job))
    return records


def load_batch(job_dir: str | Path, attempt: int) -> list[dict[str, Any]]:
    """Load one attempt batch from the saved collection summary.

    A batch contains at most one trial per collected problem, with the same attempt
    index. For example, 70 problems run five times produce five 70-trial batches;
    attempt 2 selects the third. This is not an API batch or a concurrency group.
    Benchmark-specific submission checks determine whether all required problems
    are present and their answers are usable.
    """
    if type(attempt) is not int or attempt < 0:
        raise ValueError("attempt must be a nonnegative integer")
    batches = read_json(Path(job_dir) / "summary.json").get("batches")
    if not isinstance(batches, list) or any(not isinstance(b, dict) for b in batches):
        raise TypeError("summary batches must be a list of objects")
    selected = [b for b in batches if b.get("attempt") == attempt]
    if len(selected) != 1 or not isinstance(selected[0].get("trials"), list):
        raise ValueError(f"summary has no unique attempt {attempt}")
    trials = selected[0]["trials"]
    if any(not isinstance(t, dict) for t in trials):
        raise TypeError("summary trials must be objects")
    if any(
        type(t.get("attempt")) is not int or t["attempt"] != attempt for t in trials
    ):
        raise ValueError("submission cannot mix attempt indices")
    ids = [t.get("problem_id") for t in trials]
    if any(not isinstance(pid, str) or not pid for pid in ids):
        raise TypeError("summary trials require problem IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("summary problem IDs must be unique")
    return trials


def collect_trials(job_dir: str | Path) -> dict[str, Any]:
    """Retain all trials in attempt batches, including incomplete selections."""
    job = Path(job_dir)
    trials = load_trials(job)
    if not trials:
        raise ValueError(f"no trial results found in {job}")
    identities = {
        (
            trial.benchmark,
            trial.agent,
            trial.model_provider,
            trial.model,
            json.dumps(trial.benchmark_config, sort_keys=True),
        )
        for trial in trials
    }
    if len(identities) != 1:
        raise ValueError(
            "cannot combine multiple benchmarks, agents, models, or configurations"
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
        attempts = sorted(set.union(*attempt_sets))
    else:
        attempts = list(range(max(map(len, groups.values()))))
    batches = []
    rows = []
    for attempt in attempts:
        batch = []
        for problem_id in sorted(groups):
            if all(explicit):
                trial = next(
                    (item for item in groups[problem_id] if item.attempt == attempt),
                    None,
                )
            else:
                if attempt >= len(groups[problem_id]):
                    continue
                trial = replace(groups[problem_id][attempt], attempt=attempt)
            if trial is None:
                continue
            batch.append(trial)
        rows.extend(batch)
        batches.append(
            {
                "attempt": attempt,
                "complete": len(batch) == len(groups),
                "validated": sum(trial.status == "validated" for trial in batch),
                "passed": sum(trial.reward == 1 for trial in batch),
                "total": len(batch),
                "trials": [asdict(trial) for trial in batch],
            }
        )

    summary = {
        "problems": len(groups),
        "trials": len(trials),
        # Completeness is relative to collected problems, not official coverage.
        "complete_batches": sum(batch["complete"] for batch in batches),
        "incomplete_batches": sum(not batch["complete"] for batch in batches),
        "unbatched_trials": len(trials) - len(rows),
        "batches": batches,
    }
    summarize = summarizer(trials[0].benchmark)
    if summarize is not None and rows:
        summary["metrics"] = summarize([asdict(trial) for trial in rows])
    write_json(job / "summary.json", summary)
    with (job / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        fields = tuple(Trial.__dataclass_fields__)
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for trial in rows:
            row = asdict(trial)
            row["benchmark_config"] = json.dumps(trial.benchmark_config, sort_keys=True)
            row["error"] = json.dumps(trial.error, sort_keys=True)
            writer.writerow(row)
    return summary
