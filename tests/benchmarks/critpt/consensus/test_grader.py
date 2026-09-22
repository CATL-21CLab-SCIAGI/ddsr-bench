from pathlib import Path

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus.bundle import digest
from ddsr_bench.benchmarks.critpt.evaluation.consensus.grader import (
    Grader,
    summarize,
)
from ddsr_bench.benchmarks.critpt.evaluation.consensus.runtime import Runtime
from ddsr_bench.benchmarks.critpt.evaluation.consensus.validation import validate


def row(number, refs, template="def answer():\n    pass"):
    return {
        "id": f"Challenge_{number}_main",
        "number": number,
        "mode": "function" if number == 45 else "value",
        "confidence": 0.4,
        "template": template,
        "template_sha256": digest(template),
        "parameters": ["mc_x", "mc_y", "mv_x", "mv_y"] if number == 45 else [],
        "reference_coverage": "complete",
        "references": [
            {"id": str(i), "group": str(i), "code": code, "sha256": digest(code)}
            for i, code in enumerate(refs)
        ],
    }


def test_tied_references_require_one_whole_function():
    template = "def answer(mc_x, mc_y, mv_x, mv_y):\n    pass"
    refs = [
        template.replace("pass", "return True"),
        template.replace("pass", "return False"),
    ]
    candidate = template.replace("pass", "return mc_x * mv_y != mc_y * mv_x")
    p = row(45, refs, template)
    ev = Grader({"problems": [p]}, Runtime(trusted_local=True, timeout=15))
    assert ev.grade(p["id"], candidate)["status"] == "different"
    assert ev.grade(p["id"], refs[1])["status"] == "matched"


def test_candidate_and_reference_failures_are_separate():
    p = row(1, ["def answer():\n    return 1"])
    ev = Grader({"problems": [p]}, Runtime(trusted_local=True, timeout=15))
    assert (
        ev.grade(p["id"], "def answer():\n    return 1/0")["status"]
        == "candidate_error"
    )
    assert (
        ev.grade(p["id"], "def wrong():\n    return 1")["status"] == "candidate_error"
    )
    bad = row(1, ["def answer():\n    return 1/0"])
    ev = Grader({"problems": [bad]}, Runtime(trusted_local=True, timeout=15))
    assert (
        ev.grade(bad["id"], "def answer():\n    return 1")["status"]
        == "reference_error"
    )


def test_subprocess_timeout_is_bounded():
    runtime = Runtime(trusted_local=True, timeout=1)
    result = runtime.run(
        {
            "action": "evaluate",
            "code": "def answer():\n    while True: pass",
            "template": "def answer():\n    pass",
            "inputs": [{}],
        }
    )
    assert result["stage"] == "timeout"


def test_skip_never_invokes_execution_or_produces_reward():
    class NoRun:
        def run(self, _):
            raise AssertionError("skip executed")

    p = {**row(25, []), "mode": "skip"}
    result = Grader({"problems": [p]}, NoRun()).grade(p["id"], "nonsense")
    assert result["status"] == "skipped"
    assert result["matched"] is None
    assert "reward" not in result and "verified" not in result


def test_summary_fixed_denominator_and_uncertainty():
    states = [
        ("matched", True),
        ("different", False),
        ("unknown", None),
        ("missing_candidate", False),
        ("skipped", None),
    ]
    results = [
        {
            "problem_id": str(i),
            "status": status,
            "matched": matched,
            "confidence": 0.8,
            "methods": ["sampled"] if i == 0 else [],
            "reference_coverage": "complete",
        }
        for i, (status, matched) in enumerate(states)
    ]
    report = summarize(results)
    assert report["active"] == 4 and report["skipped"] == 1
    assert report["match_rate"] == 0.25 and report["weighted_match_rate"] == 0.25
    assert report["sampled_matches"] == 1


@pytest.mark.parametrize("number", [47, 51])
def test_audited_exclusions_keep_reason_without_executing_bad_candidates(
    number, sample_bundle
):
    class NoRun:
        def run(self, _):
            raise AssertionError("excluded candidate or reference executed")

    grader = Grader(sample_bundle, NoRun())
    result = grader.grade(
        f"Challenge_{number}_main", "nonsense", input_error={"stage": "generation"}
    )
    assert result["status"] == "skipped" and result["matched"] is None
    assert "2026-09-17 audit" in result["skip_reason"]
    summary = summarize([result])
    assert summary["active"] == 0 and summary["weight_total"] == 0
    assert summary["match_rate"] is None


def test_validator_does_not_accept_import_or_signature_changes():
    with pytest.raises(ValueError):
        validate("import os\ndef answer(): return 1", "def answer(): pass")
    with pytest.raises(ValueError):
        validate("def answer(x): return x", "def answer(): pass")


def test_consensus_does_not_import_reward_graders():
    from ddsr_bench.benchmarks.critpt.evaluation import consensus

    root = Path(consensus.__file__).parent
    assert list(root.glob("*.py"))
    for path in root.glob("*.py"):
        text = path.read_text()
        for module in ("compare", "numeric", "symbolic", "verifier"):
            assert f"ddsr_bench.grading.{module}" not in text


def test_nonfinite_reference_is_not_a_matching_target():
    p = row(1, ["def answer():\n    return float('nan')"])
    ev = Grader({"problems": [p]}, Runtime(trusted_local=True, timeout=15))
    assert (
        ev.grade(p["id"], "def answer():\n    return float('nan')")["status"]
        == "reference_error"
    )


def test_execution_reports_runtime_and_discards_candidate_prints():
    runtime = Runtime(trusted_local=True, timeout=15)
    result = runtime.run(
        {
            "action": "evaluate",
            "code": "def answer():\n    print('not JSON')\n    return 3",
            "template": "def answer():\n    pass",
            "inputs": [{}],
        }
    )
    assert result["status"] == "ok"
    assert result["outputs"][0]["v"] == "3"
    assert result["runtime"]["dependencies"]["sympy"]


def test_default_execution_does_not_silently_fall_back_to_host(monkeypatch):
    from ddsr_bench.benchmarks.critpt.evaluation.consensus import runtime

    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Docker is required"):
        Runtime()


def test_docker_worker_is_pinned_and_has_no_reference_mounts(monkeypatch):
    import subprocess

    from ddsr_bench.benchmarks.critpt.evaluation.consensus import runtime

    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, stdout="sha256:abc\n"),
    )
    seen = {}

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            seen["command"] = command
            seen["request"] = kwargs["stdin"].read()
            self.stdout = kwargs["stdout"]

        def wait(self, **kwargs):
            self.stdout.write(b'{"status":"ok"}')
            self.stdout.flush()

    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    executor = Runtime()
    assert (
        executor.run({"action": "evaluate", "code": "private-candidate"})["status"]
        == "ok"
    )
    command = seen["command"]
    assert "--network=none" in command and "--read-only" in command
    assert "--pull=never" in command and "sha256:abc" in command
    assert "--user=65534:65534" in command
    assert "-v" not in command and "--volume" not in command
    assert b"private-candidate" in seen["request"]
    assert "private-candidate" not in command


def test_large_request_reaches_worker_without_stalling():
    result = Runtime(trusted_local=True, timeout=10).run(
        {
            "action": "evaluate",
            "code": "# " + "padding" * 150_000 + "\ndef answer(): return 7",
            "template": "def answer(): pass",
            "inputs": [{}],
        }
    )
    assert result["status"] == "ok"
    assert result["outputs"][0]["v"] == "7"
