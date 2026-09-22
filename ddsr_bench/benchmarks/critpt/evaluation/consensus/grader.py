"""Consensus-only results: never produce legacy reward/verified fields."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from threading import Lock

from . import POLICY_VERSION
from .bundle import digest
from .candidates import Candidate, CandidateBatch
from .fixtures import inputs
from .runtime import Runtime
from .wire import encode


class Grader:
    """Match candidates against approved references, caching bounded executions."""

    def __init__(self, bundle: dict, runtime: Runtime):
        self.rows = {p["id"]: p for p in bundle["problems"]}
        self.runtime = runtime
        self.cache = {}
        self.lock = Lock()

    def execution(self, code, row):
        cases = [
            {k: encode(v) for k, v in c.items()}
            for c in inputs(row["number"], row["template"])
        ]
        key = digest(json.dumps([code, row["template_sha256"], cases], sort_keys=True))
        with self.lock:
            cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self.runtime.run(
            {
                "action": "evaluate",
                "code": code,
                "template": row["template"],
                "inputs": cases,
            }
        )
        with self.lock:
            self.cache[key] = result
        return result

    def grade(
        self, problem_id: str, code: str | None, *, input_error: dict | None = None
    ) -> dict:
        row = self.rows[problem_id]
        base = {
            "problem_id": problem_id,
            "evaluation_kind": "internal_consensus",
            "policy_version": POLICY_VERSION,
            "confidence": row["confidence"],
            "confidence_basis": "historical_report_not_recomputed",
            "mode": row["mode"],
            "reference_coverage": row["reference_coverage"],
            "status": "skipped",
            "matched": None,
            "methods": [],
        }
        if row["mode"] == "skip":
            return {**base, "skip_reason": row.get("note", "")}
        if input_error is not None:
            return {
                **base,
                "status": "candidate_error",
                "matched": False,
                "error": input_error,
            }
        if code is None:
            return {**base, "status": "missing_candidate", "matched": False}
        ready, errors = [], []
        # Each reference is executed alone. Candidate receives no reference code.
        for ref in row["references"]:
            result = self.execution(ref["code"], row)
            if result["status"] == "ok":
                ready.append(
                    {
                        "id": ref["id"],
                        "group": ref["group"],
                        "outputs": result["outputs"],
                    }
                )
            else:
                errors.append({"reference": ref["id"], **result})
        if not ready:
            return {**base, "status": "reference_error", "reference_errors": errors}
        candidate = self.execution(code, row)
        base["runtime"] = candidate.get("runtime")
        if candidate["status"] != "ok":
            return {
                **base,
                "status": "candidate_error",
                "matched": False,
                "error": candidate,
                "reference_errors": errors,
            }
        result = self.runtime.run(
            {
                "action": "compare",
                "number": row["number"],
                "parameters": row["parameters"],
                "candidate": candidate["outputs"],
                "references": ready,
            }
        )
        if result["status"] != "ok":
            return {
                **base,
                "status": "unknown",
                "error": result,
                "reference_errors": errors,
            }
        comparisons = result["comparisons"]
        match = next((x for x in comparisons if x["status"] == "matched"), None)
        if match:
            group = next(r["group"] for r in ready if r["id"] == match["reference"])
            return {
                **base,
                "status": "matched",
                "matched": True,
                "methods": match["methods"],
                "matched_reference": match["reference"],
                "matched_group": group,
                "reference_errors": errors,
                "comparisons": comparisons,
            }
        unknown = errors or any(x["status"] == "unknown" for x in comparisons)
        return {
            **base,
            "status": "unknown" if unknown else "different",
            "matched": None if unknown else False,
            "reference_errors": errors,
            "comparisons": comparisons,
        }


def grade_candidate(
    grader: Grader, problem_id: str, candidate: Candidate | None
) -> dict:
    """Grade one policy slot and attach available candidate provenance."""
    result = grader.grade(
        problem_id,
        candidate.code if candidate else None,
        input_error=candidate.error if candidate else None,
    )
    if candidate:
        result.update(
            candidate_source=str(candidate.path),
            generation=candidate.generation,
            metadata_errors=candidate.metadata_errors,
        )
    return result


def report(directory: Path, batch: CandidateBatch, results: list[dict]) -> dict:
    """Assemble one attempt without changing result order or batch-only fields."""
    return {
        "candidate_input": {
            "directory": str(directory),
            "layout": batch.layout,
            "attempt": batch.attempt,
            "available_attempts": batch.available_attempts,
            "recognized_candidates": len(batch.answers),
        },
        "summary": summarize(results),
        "results": results,
    }


def summarize(results: list[dict]) -> dict:
    if len({r["problem_id"] for r in results}) != len(results):
        raise ValueError("duplicate problem results")
    active = [r for r in results if r["status"] != "skipped"]
    weight = sum(r["confidence"] for r in active)
    matched = sum(r["matched"] is True for r in active)
    by_confidence = {}
    for confidence in sorted({r["confidence"] for r in active}, reverse=True):
        group = [r for r in active if r["confidence"] == confidence]
        count = sum(r["matched"] is True for r in group)
        by_confidence[str(confidence)] = {
            "matched": count,
            "total": len(group),
            "match_rate": count / len(group),
        }
    return {
        "evaluation_kind": "internal_consensus",
        "policy_version": POLICY_VERSION,
        "total_slots": len(results),
        "active": len(active),
        "skipped": len(results) - len(active),
        "matched": matched,
        "match_rate": matched / len(active) if active else None,
        "weight_total": round(weight, 10),
        "weighted_match_rate": (
            sum(r["confidence"] for r in active if r["matched"] is True) / weight
            if weight
            else None
        ),
        "statuses": dict(Counter(r["status"] for r in results)),
        "sampled_matches": sum(
            r["matched"] is True and "sampled" in r["methods"] for r in active
        ),
        "incomplete_reference_coverage": [
            r["problem_id"] for r in active if r["reference_coverage"] == "incomplete"
        ],
        "by_confidence": by_confidence,
    }
