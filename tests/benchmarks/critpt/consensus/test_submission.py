import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ddsr_bench.benchmarks.collect import collect_trials, load_batch
from ddsr_bench.benchmarks.critpt.evaluation import local as submission
from ddsr_bench.benchmarks.critpt.evaluation.consensus import cli
from ddsr_bench.benchmarks.critpt.evaluation.consensus.runtime import Runtime
from ddsr_bench.commands.dispatch import dispatch


@pytest.mark.parametrize("benchmark", ["critpt", "scicode", "cmphysbench", "phybench"])
def test_config_scope(benchmark):
    configs = Path(__file__).resolve().parents[4] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(configs)):
        config = compose(config_name="config", overrides=[f"benchmark={benchmark}"])
    assert "local" not in config.submission
    submission = config.benchmark.get("submission", {})
    assert ("local" in submission) == (benchmark == "critpt")


def test_local_submission(tmp_path, sample_bundle_path, monkeypatch):
    """Both entry points must retain the same report and incomplete attempt."""
    job = tmp_path / "job"
    artifacts = job / "Challenge_1_main__attempt-0" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "answer.py").write_text("def answer(): return 1\n")
    (artifacts.parent / "result.json").write_text(
        json.dumps(
            {
                "task_name": "critpt/Challenge_1_main",
                "trial_name": artifacts.parent.name,
                "attempt": 0,
                "agent_info": {"name": "teacher", "model_info": {"name": "model"}},
                "static_result": {"status": "validated"},
            }
        )
    )
    collect_trials(job)
    # Only this reviewed test function executes locally, never model outputs.
    monkeypatch.setattr(
        submission, "Runtime", lambda **_: Runtime(trusted_local=True, timeout=15)
    )
    config = OmegaConf.create(
        {
            "action": "submit",
            "benchmark": {
                "name": "critpt",
                "submission": {"local": {"bundle": str(sample_bundle_path), "jobs": 1}},
            },
            "paths": {"input": str(job)},
            "submission": {
                "backend": "local",
                "attempt": 0,
            },
        }
    )
    official = job / "submission-0.json"
    official.write_text('{"existing": true}')

    dispatch(config)
    local = json.loads((job / "submission-local-0.json").read_text())
    old = tmp_path / "cli.json"
    assert (
        cli.main(
            [
                "score",
                "--bundle",
                str(sample_bundle_path),
                "--candidates",
                str(job),
                "--attempt",
                "0",
                "--output",
                str(old),
                "--trusted-local",
                "--jobs",
                "1",
            ]
        )
        == 0
    )
    original = json.loads(old.read_text())
    assert local["candidate_input"]["layout"] == "collected"
    local["candidate_input"]["layout"] = original["candidate_input"]["layout"]
    assert local == original
    assert local["summary"]["statuses"] == {
        "matched": 1,
        "missing_candidate": 60,
        "skipped": 9,
    }
    assert local["summary"]["match_rate"] == pytest.approx(1 / 61)
    assert json.loads(official.read_text()) == {"existing": True}
    with pytest.raises(ValueError, match="already exists"):
        dispatch(config)


def test_unknown_backend(tmp_path):
    from ddsr_bench.benchmarks.critpt.evaluation.submit import submit_attempt

    with pytest.raises(ValueError, match="unknown submission backend"):
        submit_attempt(tmp_path, 0, {}, {"backend": "typo"})


def test_collection_required(tmp_path, sample_bundle_path, monkeypatch):
    artifacts = tmp_path / "Challenge_1_main__attempt-0" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "answer.py").write_text("def answer(): return 1")
    monkeypatch.setattr(
        submission, "Runtime", lambda **_: pytest.fail("must not start execution")
    )
    with pytest.raises(ValueError, match="summary.json"):
        submission.submit(tmp_path, 0, bundle=str(sample_bundle_path))


def test_collected_names(tmp_path):
    from ddsr_bench.benchmarks.critpt.evaluation.consensus.candidates import (
        load_collected,
    )

    trial = tmp_path / "harbor-random-name"
    trial.mkdir()
    (trial / "solution.py").write_text("def answer(): return 7")
    rows = [
        {
            "problem_id": "Challenge_1_main",
            "trial_name": trial.name,
            "attempt": 2,
            "artifact": f"{trial.name}/solution.py",
        }
    ]
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"batches": [{"attempt": 2, "trials": rows}]}))
    assert (
        load_collected(tmp_path, 2).answers["Challenge_1_main"].code
        == "def answer(): return 7\n"
    )
    with pytest.raises(ValueError, match="no unique attempt"):
        load_batch(tmp_path, 0)
    rows[0]["attempt"] = 0
    summary.write_text(json.dumps({"batches": [{"attempt": 2, "trials": rows}]}))
    with pytest.raises(ValueError, match="mix attempt"):
        load_collected(tmp_path, 2)
