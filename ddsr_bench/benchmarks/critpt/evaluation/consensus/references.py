"""Build and load reviewed, verifier-only consensus references."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from ddsr_bench.benchmarks.utils import write_json
from ddsr_bench.grading.validation import extract_code, validate_code

from . import POLICY_VERSION
from .matching.rules import NOTES, mode


def checksum(text: str) -> str:
    """Return the SHA-256 hex checksum of UTF-8 text."""
    return hashlib.sha256(text.encode()).hexdigest()


def _parameter_names(template: str) -> list[str]:
    """Read the ordered answer() parameter names from a code template."""
    fn = next(
        n
        for n in ast.parse(template).body
        if isinstance(n, ast.FunctionDef) and n.name == "answer"
    )
    return [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]


def build_references(manifest_path: Path, source_root: Path, destination: Path) -> dict:
    """Build a reference file from the reviewed manifest and source answers."""
    inventory = json.loads(manifest_path.read_text())
    ids = [problem["problem_id"] for problem in inventory["problems"]]
    if set(ids) != {f"Challenge_{i}_main" for i in range(1, 71)} or len(ids) != 70:
        raise ValueError(
            "inventory must contain each of the 70 main challenges exactly once"
        )
    problems = []
    for item in inventory["problems"]:
        number = int(re.fullmatch(r"Challenge_(\d+)_main", item["problem_id"])[1])
        official = source_root / item["official_problem_path"]
        problem = next(
            problem
            for problem in json.loads(official.read_text())["problems"]
            if problem["problem_id"] == item["problem_id"]
        )
        template = problem["code_template"]
        references, groups = [], []
        for gi, group in enumerate(item["reported_groups"]):
            if not group["is_maximal_group"]:
                continue
            group_id = f"group-{gi+1}"
            group_refs = []
            for sample in group["samples"]:
                if not sample.get("path") or mode(number) == "skip":
                    continue
                path = source_root / sample["path"]
                raw = path.read_text()
                if sample.get("sha256") and checksum(raw) != sample["sha256"]:
                    raise ValueError(f"source hash changed: {path}")
                data = json.loads(raw)
                code = extract_code(data["generated_code"])
                validate_code(code, template)
                rid = f"{group_id}/{sample['reported_model']}"
                references.append(
                    {
                        "id": rid,
                        "group": group_id,
                        "code": code,
                        "sha256": checksum(code),
                        "reported_model": sample["reported_model"],
                        "file_model": data.get("model"),
                        "source_sha256": checksum(raw),
                        "provenance_status": sample.get("provenance_status"),
                    }
                )
                group_refs.append(rid)
            groups.append(
                {"id": group_id, "models": group["models"], "references": group_refs}
            )
        status = mode(number)
        if status != "skip" and not references:
            raise ValueError(f"no reference for active {item['problem_id']}")
        confidence = {2: 0.4, 3: 0.6, 4: 0.8, 1: 0.0}[item["reported_max_agreement"]]
        if confidence != item["confidence"]:
            raise ValueError("confidence does not match the reviewed historical policy")
        problems.append(
            {
                "id": item["problem_id"],
                "number": number,
                "mode": status,
                "confidence": confidence,
                "reported_max_agreement": item["reported_max_agreement"],
                "template": template,
                "template_sha256": checksum(template),
                "parameters": _parameter_names(template),
                "official_sha256": checksum(official.read_text()),
                "note": NOTES.get(number, ""),
                "reference_coverage": (
                    "incomplete"
                    if status != "skip" and any(not g["references"] for g in groups)
                    else "complete"
                ),
                "groups": groups,
                "references": references,
            }
        )
    result = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "kind": "internal_consensus_reference_bundle",
        "confidence_basis": "historical_report_not_recomputed",
        "source_manifest_sha256": checksum(manifest_path.read_text()),
        "official_commit": inventory.get("official_commit"),
        "problems": sorted(problems, key=lambda problem: problem["number"]),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result, overwrite=False, newline=True)
    return result


def load_references(path: Path) -> dict:
    """Validate the existing shared reference file and its full policy inventory."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"References not found: {path}. Supply the private JSON path via "
            "benchmark.submission.internal.references or replay --references."
        ) from error
    references = json.loads(raw)
    if (
        references.get("schema_version") != 1
        or references.get("policy_version") != POLICY_VERSION
    ):
        raise ValueError("unsupported reference schema or policy version")
    problems = references["problems"]
    if len(problems) != 70 or {problem["number"] for problem in problems} != set(
        range(1, 71)
    ):
        raise ValueError("references must preserve all 70 problem slots")
    for problem in problems:
        _validate_problem(problem)
    return references


def _validate_problem(problem: dict) -> None:
    """Validate one problem entry, including its reference answers and groups."""
    if (
        problem["number"] not in range(1, 71)
        or problem["id"] != f"Challenge_{problem['number']}_main"
        or problem["mode"] != mode(problem["number"])
    ):
        raise ValueError("reference scope does not match policy")
    if checksum(problem["template"]) != problem["template_sha256"]:
        raise ValueError("template checksum mismatch")
    if problem["parameters"] != _parameter_names(problem["template"]):
        raise ValueError("parameter inventory differs from template")
    if problem["confidence"] != {1: 0.0, 2: 0.4, 3: 0.6, 4: 0.8}.get(
        problem["reported_max_agreement"]
    ):
        raise ValueError("inconsistent reported confidence")
    if problem["mode"] != "skip" and not problem["references"]:
        raise ValueError("active problem has no references")
    ref_ids = {r["id"] for r in problem["references"]}
    if len(ref_ids) != len(problem["references"]):
        raise ValueError("duplicate reference ID")
    grouped = [rid for g in problem["groups"] for rid in g["references"]]
    if set(grouped) != ref_ids or len(grouped) != len(ref_ids):
        raise ValueError("reference group inventory is inconsistent")
    incomplete = problem["mode"] != "skip" and any(
        not g["references"] for g in problem["groups"]
    )
    if problem["reference_coverage"] != ("incomplete" if incomplete else "complete"):
        raise ValueError("incorrect reference coverage flag")
    for ref in problem["references"]:
        if checksum(ref["code"]) != ref["sha256"]:
            raise ValueError("reference checksum mismatch")
        if not any(
            g["id"] == ref["group"] and ref["id"] in g["references"]
            for g in problem["groups"]
        ):
            raise ValueError("incorrect reference group membership")
