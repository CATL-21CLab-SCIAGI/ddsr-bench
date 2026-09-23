"""Consensus-only results: never produce legacy reward/verified fields."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from threading import Lock

from ddsr_bench.benchmarks.utils import read_bytes, read_json, write_json

from . import POLICY_VERSION
from .execution.runtime import Runtime
from .execution.serialization import encode
from .matching.cases import inputs
from .references import checksum


class Grader:
    """Match answers against approved references, caching bounded executions."""

    def __init__(
        self, references: dict, runtime: Runtime, *, cache_dir: Path | None = None
    ):
        self.rows = {p["id"]: p for p in references["problems"]}
        self.runtime = runtime
        self.cache = {}
        self.lock = Lock()
        self.cache_dir = None
        self.cache_stats = {"hits": 0, "executions": 0}
        if cache_dir is not None:
            if runtime.backend != "docker" or not runtime.image_id:
                raise ValueError("reference caching requires a pinned Docker image")
            # The image ID pins worker code, validation, and scientific libraries.
            settings = [
                POLICY_VERSION,
                runtime.image_id,
                runtime.timeout,
                runtime.cpus,
                runtime.memory_mb,
            ]
            self.cache_dir = Path(cache_dir) / checksum(json.dumps(settings))

    def execution(self, code, row, *, reference=False):
        cases = [
            {k: encode(v) for k, v in c.items()}
            for c in inputs(row["number"], row["parameters"])
        ]
        payload = {
            "action": "evaluate",
            "code": code,
            "template": row["template"],
            "inputs": cases,
        }
        # Never let an answer hit a reference entry, even when its code is identical.
        if not reference:
            return self.runtime.run(payload)
        key = checksum(json.dumps(payload, sort_keys=True))
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        with self.lock:
            cached = self.cache.get(key)
        if cached is None and path is not None:
            try:
                record = read_json(path, max_bytes=8_000_000, allow_nan=False)
                value = record.get("result")
                if (
                    record.get("key") == key
                    and isinstance(value, dict)
                    and record.get("sha256")
                    == checksum(json.dumps(value, sort_keys=True))
                    and value.get("status") == "ok"
                    and isinstance(value.get("outputs"), list)
                    and len(value["outputs"]) == len(cases)
                    and isinstance(value.get("runtime"), dict)
                ):
                    cached = value
            except (OSError, ValueError, TypeError, RecursionError):
                pass  # Missing/corrupt cache entries are recomputed, never scored.
        if cached is not None:
            with self.lock:
                self.cache[key] = cached
                self.cache_stats["hits"] += 1
            return cached
        with self.lock:
            self.cache_stats["executions"] += 1
        result = self.runtime.run(payload)
        if result["status"] == "ok":
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                # Atomic private files: concurrent cold misses may safely publish
                # the same deterministic entry without exposing partial JSON.
                write_json(
                    path,
                    {
                        "key": key,
                        "result": result,
                        "sha256": checksum(json.dumps(result, sort_keys=True)),
                    },
                    allow_nan=False,
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
                "status": "answer_error",
                "matched": False,
                "error": input_error,
            }
        if code is None:
            return {**base, "status": "missing_answer", "matched": False}
        ready, errors = [], []
        # Each reference executes alone; generated answers receive no reference code.
        for ref in row["references"]:
            result = self.execution(ref["code"], row, reference=True)
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
        answer = self.execution(code, row)
        base["runtime"] = answer.get("runtime")
        if answer["status"] != "ok":
            return {
                **base,
                "status": "answer_error",
                "matched": False,
                "error": answer,
                "reference_errors": errors,
            }
        result = self.runtime.run(
            {
                "action": "compare",
                "number": row["number"],
                "parameters": row["parameters"],
                "answer": answer["outputs"],
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


def grade(
    grader: Grader,
    problem_id: str,
    path: Path | None,
    status: str | None,
    *,
    upstream: dict | None = None,
) -> dict:
    """Read an already extracted answer; execution stays in the grader's sandbox.

    Failure details come from shared collection, not historical response loading.
    """
    code, error = None, None
    if path is not None:
        try:
            code = read_bytes(path, max_bytes=8_000_000).decode("utf-8")
            if not code.strip():
                raise ValueError("empty answer")
        except (OSError, ValueError) as failure:
            error = {
                "status": "error",
                "stage": "input",
                "error": f"{type(failure).__name__}: {failure}"[:300],
            }
    elif status == "error":
        error = {
            "status": "error",
            "stage": "generation",
            "error": "rollout did not produce an answer artifact",
        }
        if upstream is not None:
            error["upstream"] = upstream
    result = grader.grade(problem_id, code, input_error=error)
    if path is not None:
        # Preserve report fields without reconstructing historical answer records.
        result.update(answer_source=str(path), generation=None, metadata_errors=[])
    return result


def summarize(results: list[dict]) -> dict:
    """Count each problem-attempt once; missing/error/unknown stay in the denominator."""
    if len({(r.get("attempt"), r["problem_id"]) for r in results}) != len(results):
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
        "incomplete_reference_coverage": list(
            dict.fromkeys(
                r["problem_id"]
                for r in active
                if r["reference_coverage"] == "incomplete"
            )
        ),
        "by_confidence": by_confidence,
    }
