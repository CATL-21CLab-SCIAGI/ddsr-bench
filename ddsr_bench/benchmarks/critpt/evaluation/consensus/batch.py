"""Score every native attempt, persisting each result before aggregating means."""

from __future__ import annotations

import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from ddsr_bench.benchmarks.utils import write_json

from .candidates import TRIAL, candidate_digest, load_candidates
from .grader import Grader, grade_candidate, report


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, data, allow_nan=False, newline=True)


def score_batch(
    directory: Path,
    output: Path,
    grader: Grader,
    *,
    jobs: int,
    metadata: dict,
    model_label: str | None = None,
) -> dict:
    """Average over all policy slots and attempts; missing/error/unknown score zero."""
    if not directory.is_dir():
        raise ValueError("batch requires one native rollout job directory")
    attempts = sorted(
        {
            int(match[2])
            for path in directory.iterdir()
            if path.is_dir() and (match := TRIAL.fullmatch(path.name))
        }
    )
    if not attempts:
        raise ValueError("batch requires native rollout trials")
    if not 1 <= jobs <= 16:
        raise ValueError("jobs must be between 1 and 16")
    batches = [load_candidates(directory, attempt) for attempt in attempts]
    output.mkdir(parents=True, exist_ok=False)
    model = model_label or directory.name
    rows = list(grader.rows.values())
    total = len(rows) * len(attempts)
    metadata = {
        **metadata,
        "candidate_directory": str(directory),
        "model": model,
        "attempts": attempts,
        "jobs": jobs,
        "mean_definition": "matched / (active policy problems * attempts); errors, missing and unknown count as zero",
        "started_at": datetime.now(UTC).isoformat(),
    }
    _save(output / "manifest.json", metadata)
    all_results, summaries = [], []
    for batch in batches:

        def score(row, batch=batch):
            candidate = batch.answers.get(row["id"])
            result = grade_candidate(grader, row["id"], candidate)
            result.update(model=model, attempt=batch.attempt)
            if candidate:
                result["candidate_sha256"] = candidate_digest(candidate.path)
            return result

        results = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(score, row) for row in rows]
            for future in as_completed(futures):
                result = future.result()
                name = f"{result['problem_id']}__attempt-{batch.attempt}.json"
                _save(output / "trials" / name, result)
                results.append(result)
                all_results.append(result)
                progress = {
                    "status": "running",
                    "completed": len(all_results),
                    "total": total,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
                _save(output / "progress.json", progress)
                if len(all_results) % 10 == 0:
                    print(json.dumps(progress), flush=True)
        results.sort(key=lambda r: int(r["problem_id"].split("_")[1]))
        attempt_report = report(directory, batch, results)
        summary = attempt_report["summary"]
        summaries.append({"attempt": batch.attempt, **summary})
        _save(
            output / f"attempt-{batch.attempt}.json",
            {"provenance": metadata, **attempt_report},
        )
    matched = sum(r["matched"] is True for r in all_results)
    active = sum(a["active"] for a in summaries)
    problems = []
    for row in rows:
        own = [r for r in all_results if r["problem_id"] == row["id"]]
        problems.append(
            {
                "problem_id": row["id"],
                **(
                    {"skip_reason": row.get("note", "")}
                    if row["mode"] == "skip"
                    else {}
                ),
                "mean_match_rate": (
                    None
                    if row["mode"] == "skip"
                    else sum(r["matched"] is True for r in own) / len(attempts)
                ),
                "attempts": [
                    {key: r[key] for key in ("attempt", "status", "matched")}
                    for r in sorted(own, key=lambda r: r["attempt"])
                ],
            }
        )
    summary = {
        "evaluation_kind": "internal_consensus",
        "total_slots": total,
        "active_trials": active,
        "skipped_trials": total - active,
        "matched_trials": matched,
        "mean_match_rate": matched / active if active else None,
        "mean_weighted_match_rate": (
            sum(a["weighted_match_rate"] for a in summaries) / len(attempts)
            if active
            else None
        ),
        "statuses": dict(Counter(r["status"] for r in all_results)),
        "attempts": summaries,
        "problems": problems,
    }
    with (output / "attempt-results.csv").open("x", newline="", encoding="utf-8") as f:
        fields = [
            "model",
            "problem_id",
            "attempt",
            "status",
            "matched",
            "score",
            "error_stage",
            "skip_reason",
            "candidate_source",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in sorted(
            all_results,
            key=lambda r: (int(r["problem_id"].split("_")[1]), r["attempt"]),
        ):
            writer.writerow(
                {
                    **{key: result[key] for key in fields if key in result},
                    "score": (
                        ""
                        if result["status"] == "skipped"
                        else int(result["matched"] is True)
                    ),
                    "error_stage": result.get("error", {}).get("stage", ""),
                }
            )
    _save(output / "summary.json", {"provenance": metadata, **summary})
    _save(
        output / "progress.json",
        {"status": "completed", "completed": total, "total": total},
    )
    return summary
