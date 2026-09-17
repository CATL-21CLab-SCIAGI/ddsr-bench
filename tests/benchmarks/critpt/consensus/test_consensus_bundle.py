import json
from copy import deepcopy

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus import POLICY_VERSION
from ddsr_bench.benchmarks.critpt.evaluation.consensus.bundle import (
    DEFAULT_BUNDLE,
    digest,
    load,
)
from ddsr_bench.benchmarks.critpt.evaluation.consensus.candidates import load_candidates
from ddsr_bench.benchmarks.critpt.evaluation.consensus.cli import parser
from ddsr_bench.benchmarks.critpt.evaluation.consensus.policy import mode


def bundle_fixture():
    rows = []
    for n in range(1, 71):
        active = mode(n) != "skip"
        template = "def answer():\n    pass"
        code = "def answer():\n    return 1"
        refs = (
            [{"id": "g/model", "group": "g", "code": code, "sha256": digest(code)}]
            if active
            else []
        )
        groups = [
            {
                "id": "g",
                "models": ["one", "two"],
                "references": ["g/model"] if active else [],
            }
        ]
        if n == 4:
            groups.append(
                {"id": "missing", "models": ["three", "four"], "references": []}
            )
        rows.append(
            {
                "id": f"Challenge_{n}_main",
                "number": n,
                "mode": mode(n),
                "confidence": 0.4,
                "reported_max_agreement": 2,
                "template": template,
                "template_sha256": digest(template),
                "parameters": [],
                "references": refs,
                "groups": groups,
                "reference_coverage": "incomplete" if n == 4 else "complete",
            }
        )
    return {"schema_version": 1, "policy_version": POLICY_VERSION, "problems": rows}


def test_bundle_checks_scope_reference_hash_and_coverage(tmp_path):
    bundle = bundle_fixture()
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    assert len(load(path)["problems"]) == 70
    mutations = [
        lambda p: p["references"][0].update(code="def answer(): return 2"),
        lambda p: p.update(confidence=0.8),
        lambda p: p.update(reference_coverage="complete"),
        lambda p: p.update(parameters=["unexpected"]),
    ]
    for change in mutations:
        changed = deepcopy(bundle)
        change(changed["problems"][3])
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            load(path)


def test_candidate_formats_and_duplicate_id_detection(tmp_path):
    p = tmp_path / "Challenge_1_main.json"
    p.write_text(
        json.dumps(
            {
                "problem_id": "Challenge_1_main",
                "generated_code": "def answer(): return 1",
            }
        )
    )
    assert (
        load_candidates(tmp_path).answers["Challenge_1_main"].code
        == "def answer(): return 1\n"
    )
    (tmp_path / "Challenge_1_main.py").write_text("def answer(): return 2")
    with pytest.raises(ValueError, match="multiple candidate"):
        load_candidates(tmp_path)


def test_shipped_bundle_is_complete_and_default_works_outside_repo(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    for command in ("score", "replay"):
        argv = [command, "--output", "result.json"]
        if command == "score":
            argv += ["--candidates", "answers"]
        args = parser().parse_args(argv)
        assert args.bundle == DEFAULT_BUNDLE
        bundle = load(args.bundle)
        rows = bundle["problems"]
        assert len(rows) == 70
        assert sum(p["mode"] != "skip" for p in rows) == 61
        assert sum(len(p["references"]) for p in rows) == 204
        assert digest(args.bundle.read_text()) == (
            "6b9edc5057a92148701bed69afa3b4fb121da89223c6978dc01ddca7005a44bc"
        )
        assert parser().parse_args(argv + ["--bundle", "custom.json"]).bundle.name == (
            "custom.json"
        )
