import json
from copy import deepcopy

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus.bundle import (
    DEFAULT_BUNDLE,
    digest,
    load,
)
from ddsr_bench.benchmarks.critpt.evaluation.consensus.candidates import load_candidates
from ddsr_bench.benchmarks.critpt.evaluation.consensus.cli import main, parser


def test_bundle_checks_scope_reference_hash_and_coverage(tmp_path, sample_bundle):
    bundle = sample_bundle
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


def test_external_bundle_is_complete_and_default_works_outside_repo(
    tmp_path, monkeypatch, reviewed_bundle
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


def test_default_bundle_checksum_rejects_changed_asset(monkeypatch, sample_bundle_path):
    from ddsr_bench.benchmarks.critpt.evaluation.consensus import bundle

    monkeypatch.setattr(bundle, "DEFAULT_BUNDLE", sample_bundle_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load(sample_bundle_path)


def test_missing_external_bundle_fails_without_creating_results(tmp_path, capsys):
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "replay",
                "--bundle",
                str(tmp_path / "missing.json"),
                "--output",
                str(output),
                "--trusted-local",
            ]
        )
        == 2
    )
    assert "--bundle /absolute/path/to/bundle.json" in capsys.readouterr().err
    assert not output.exists()
