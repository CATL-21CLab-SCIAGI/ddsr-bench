"""Internal submission using the existing consensus reference policy."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ddsr_bench.benchmarks.collect import load_batch

from ..evaluation.consensus.execution.runtime import Runtime, provenance
from ..evaluation.consensus.grader import Grader, grade, summarize
from ..evaluation.consensus.references import load_references
from .official import OFFICIAL_IDS


def _artifact(job: Path, trial: dict) -> Path | None:
    """Resolve a collected artifact without rediscovering generation outputs."""
    if (
        trial.get("benchmark", "critpt") != "critpt"
        or trial["problem_id"] not in OFFICIAL_IDS
    ):
        raise ValueError("internal submission requires CritPt main problems")
    artifact = trial.get("artifact")
    if artifact is None:
        return None
    if not isinstance(artifact, str):
        raise TypeError("summary artifact must be a path or null")
    path = job / artifact
    if not path.resolve().is_relative_to(job.resolve()):
        raise ValueError("summary artifact must stay inside its job")
    return path


def submit_internal(
    job: Path,
    attempts: list[int],
    *,
    references: str,
    execution: str = "docker",
    image: str = "ddsr-bench-critpt:latest",
    timeout: float = 60,
    jobs: int = 4,
    cpus: int = 2,
    memory_mb: int = 4096,
    cache_dir: str | None = None,
) -> dict:
    """Grade collected artifacts using isolated workers, without an agent trial."""
    if not 1 <= jobs <= 16:
        raise ValueError("jobs must be between 1 and 16")
    if not isinstance(references, str) or not references.strip():
        raise ValueError("internal submission requires an explicit references path")
    path = Path(references)
    reference_data = load_references(path)
    # Preflight every selection before starting any isolated execution.
    batches = [load_batch(job, attempt) for attempt in attempts]
    artifacts = {
        (trial["attempt"], trial["problem_id"]): _artifact(job, trial)
        for trials in batches
        for trial in trials
    }
    runtime = Runtime(
        backend=execution, image=image, timeout=timeout, cpus=cpus, memory_mb=memory_mb
    )
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None and cache.resolve().is_relative_to(job.resolve()):
        raise ValueError("reference cache must be outside the generation job")
    grader = Grader(reference_data, runtime, cache_dir=cache)

    results, summaries = [], []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # Finish each attempt before the next; keep one grader/cache and pool.
        for attempt, trials in zip(attempts, batches):
            collected = {trial["problem_id"]: trial for trial in trials}

            def grade_problem(problem, attempt=attempt, collected=collected):
                problem_id = problem["id"]
                trial = collected.get(problem_id, {})
                result = grade(
                    grader,
                    problem_id,
                    artifacts.get((attempt, problem_id)),
                    trial.get("status"),
                    upstream=trial.get("error"),
                )
                return result | {"attempt": attempt}

            batch = list(pool.map(grade_problem, grader.rows.values()))
            summaries.append({"attempt": attempt, **summarize(batch)})
            results.extend(batch)
    summary = summarize(results)
    summary["attempts"] = summaries
    summary["problems"] = []
    for problem in grader.rows.values():
        own = [result for result in results if result["problem_id"] == problem["id"]]
        summary["problems"].append(
            {
                "problem_id": problem["id"],
                "mean_match_rate": (
                    None
                    if problem["mode"] == "skip"
                    else sum(result["matched"] is True for result in own)
                    / len(attempts)
                ),
                "attempts": [
                    {key: result[key] for key in ("attempt", "status", "matched")}
                    for result in own
                ],
            }
        )
    return {
        "provenance": provenance(path, runtime),
        "reference_cache": grader.cache_stats,
        "answer_input": {
            "directory": str(job),
            "layout": "collected",
            "attempt": attempts[0] if len(attempts) == 1 else None,
            "available_attempts": attempts,
            "recognized_answers": sum(map(len, batches)),
        },
        "summary": summary,
        "results": results,
    }
