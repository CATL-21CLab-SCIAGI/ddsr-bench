"""Import reviewed consensus inventory into a local, verifier-only bundle."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ddsr_bench.benchmarks.utils import write_json

from . import POLICY_VERSION
from .policy import NOTES, mode
from .validation import parameters, source, validate

DEFAULT_BUNDLE = (
    Path("/mnt/workspace/zhizhou/assets/critpt/references") / f"{POLICY_VERSION}.json"
)
DEFAULT_BUNDLE_SHA256 = (
    "6b9edc5057a92148701bed69afa3b4fb121da89223c6978dc01ddca7005a44bc"
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build(manifest_path: Path, source_root: Path, destination: Path) -> dict:
    inventory = json.loads(manifest_path.read_text())
    ids = [p["problem_id"] for p in inventory["problems"]]
    if set(ids) != {f"Challenge_{i}_main" for i in range(1, 71)} or len(ids) != 70:
        raise ValueError(
            "inventory must contain each of the 70 main challenges exactly once"
        )
    rows = []
    for item in inventory["problems"]:
        number = int(re.fullmatch(r"Challenge_(\d+)_main", item["problem_id"])[1])
        official = source_root / item["official_problem_path"]
        problem = next(
            p
            for p in json.loads(official.read_text())["problems"]
            if p["problem_id"] == item["problem_id"]
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
                if sample.get("sha256") and digest(raw) != sample["sha256"]:
                    raise ValueError(f"source hash changed: {path}")
                data = json.loads(raw)
                code = source(data["generated_code"])
                validate(code, template)
                rid = f"{group_id}/{sample['reported_model']}"
                references.append(
                    {
                        "id": rid,
                        "group": group_id,
                        "code": code,
                        "sha256": digest(code),
                        "reported_model": sample["reported_model"],
                        "file_model": data.get("model"),
                        "source_sha256": digest(raw),
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
        rows.append(
            {
                "id": item["problem_id"],
                "number": number,
                "mode": status,
                "confidence": confidence,
                "reported_max_agreement": item["reported_max_agreement"],
                "template": template,
                "template_sha256": digest(template),
                "parameters": parameters(template),
                "official_sha256": digest(official.read_text()),
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
        "source_manifest_sha256": digest(manifest_path.read_text()),
        "official_commit": inventory.get("official_commit"),
        "problems": sorted(rows, key=lambda p: p["number"]),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result, overwrite=False, newline=True)
    return result


def load(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Reference bundle not found: {path}. Reference data is stored outside "
            "the repository; provide its location with --bundle /absolute/path/to/bundle.json."
        ) from error
    if (
        path.resolve() == DEFAULT_BUNDLE.resolve()
        and digest(raw) != DEFAULT_BUNDLE_SHA256
    ):
        raise ValueError("default reference bundle SHA-256 mismatch")
    bundle = json.loads(raw)
    if (
        bundle.get("schema_version") != 1
        or bundle.get("policy_version") != POLICY_VERSION
    ):
        raise ValueError("unsupported bundle or policy version")
    rows = bundle["problems"]
    if len(rows) != 70 or {p["number"] for p in rows} != set(range(1, 71)):
        raise ValueError("bundle must preserve all 70 problem slots")
    for p in rows:
        if p["id"] != f"Challenge_{p['number']}_main" or p["mode"] != mode(p["number"]):
            raise ValueError("bundle scope does not match policy")
        if digest(p["template"]) != p["template_sha256"]:
            raise ValueError("template checksum mismatch")
        if p["parameters"] != parameters(p["template"]):
            raise ValueError("parameter inventory differs from template")
        if p["confidence"] != {1: 0.0, 2: 0.4, 3: 0.6, 4: 0.8}.get(
            p["reported_max_agreement"]
        ):
            raise ValueError("inconsistent reported confidence")
        if p["mode"] != "skip" and not p["references"]:
            raise ValueError("active problem has no references")
        ref_ids = {r["id"] for r in p["references"]}
        if len(ref_ids) != len(p["references"]):
            raise ValueError("duplicate reference ID")
        grouped = [rid for g in p["groups"] for rid in g["references"]]
        if set(grouped) != ref_ids or len(grouped) != len(ref_ids):
            raise ValueError("reference group inventory is inconsistent")
        incomplete = p["mode"] != "skip" and any(
            not g["references"] for g in p["groups"]
        )
        if p["reference_coverage"] != ("incomplete" if incomplete else "complete"):
            raise ValueError("incorrect reference coverage flag")
        for ref in p["references"]:
            if digest(ref["code"]) != ref["sha256"]:
                raise ValueError("reference checksum mismatch")
            if not any(
                g["id"] == ref["group"] and ref["id"] in g["references"]
                for g in p["groups"]
            ):
                raise ValueError("incorrect reference group membership")
    return bundle
