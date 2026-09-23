import json
from pathlib import Path
from threading import Barrier, get_ident
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from ddsr_bench.benchmarks.collect import collect_trials, load_batch
from ddsr_bench.benchmarks.critpt import submission as submitter
from ddsr_bench.benchmarks.critpt.evaluation.consensus.execution.runtime import (
    DiagnosticRuntime as Runtime,
)
from ddsr_bench.benchmarks.critpt.submission import internal as submission
from ddsr_bench.commands.dispatch import dispatch


@pytest.mark.parametrize("benchmark", ["critpt", "scicode", "cmphysbench", "phybench"])
def test_config_scope(benchmark):
    configs = Path(__file__).resolve().parents[4] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(configs)):
        config = compose(config_name="config", overrides=[f"benchmark={benchmark}"])
    assert "submission" not in config
    submission = config.benchmark.get("submission", {})
    assert ("internal" in submission) == (benchmark == "critpt")
    if benchmark == "critpt":
        assert OmegaConf.is_missing(submission.internal, "references")


@pytest.mark.parametrize("backend", ["official", "internal"])
def test_required_config(tmp_path, monkeypatch, backend):
    configs = Path(__file__).resolve().parents[4] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(configs)):
        config = compose(
            config_name="config",
            overrides=[
                "action=submit",
                "benchmark=critpt",
                f"paths.input={tmp_path}",
                "benchmark.submission.attempts=0",
                f"benchmark.submission.backend={backend}",
            ],
        )
    monkeypatch.setenv("ARTIFICIAL_ANALYSIS_API_KEY", "fake")
    monkeypatch.setattr(submitter, "collect_trials", lambda _: {})
    monkeypatch.setattr(submitter, "build_batch", lambda *args: {})
    monkeypatch.setattr(submitter, "submit_official", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        submission,
        "submit_internal",
        lambda *args, **kwargs: pytest.fail("must not grade"),
    )
    if backend == "internal":
        with pytest.raises(MissingMandatoryValue, match="references"):
            dispatch(config)
        assert not list(tmp_path.iterdir())
    else:
        # An unused required internal setting must not block official submission.
        dispatch(config)
        assert (tmp_path / "submission-0.json").is_file()


@pytest.mark.parametrize("collected", [False, True])
@pytest.mark.parametrize("attempt", [0, 1])
def test_internal_submission(
    tmp_path, sample_bundle_path, monkeypatch, collected, attempt
):
    """Collected artifacts retain the fixed policy denominator and saved results."""
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
    if attempt == 1:
        record = json.loads((artifacts.parent / "result.json").read_text())
        for name, problem_id, index in [
            ("first-extra", "Challenge_1_main", 1),
            ("second", "Challenge_2_main", 0),
        ]:
            directory = job / name
            directory.mkdir()
            (directory / "result.json").write_text(
                json.dumps(
                    record
                    | {
                        "task_name": f"critpt/{problem_id}",
                        "trial_name": name,
                        "attempt": index,
                    }
                )
            )
        copied = job / "first-extra" / "artifacts"
        copied.mkdir()
        (copied / "answer.py").write_bytes((artifacts / "answer.py").read_bytes())
    if collected:
        collect_trials(job)
        before = (job / "summary.json").read_bytes()
        monkeypatch.setattr(
            submitter, "collect_trials", lambda _: pytest.fail("already collected")
        )
    # Only this reviewed test function executes locally, never model outputs.
    monkeypatch.setattr(
        submission, "Runtime", lambda **_: Runtime(trusted_local=True, timeout=15)
    )
    config = OmegaConf.create(
        {
            "action": "submit",
            "benchmark": {
                "name": "critpt",
                "submission": {
                    "backend": "internal",
                    "attempts": [attempt],
                    "internal": {"references": str(sample_bundle_path), "jobs": 1},
                },
            },
            "paths": {"input": str(job)},
        }
    )
    official = job / "submission-0.json"
    official.write_text('{"existing": true}')

    dispatch(config)
    assert (job / "summary.csv").is_file()
    assert load_batch(job, attempt)[0]["problem_id"] == "Challenge_1_main"
    if collected:
        assert (job / "summary.json").read_bytes() == before
    internal = json.loads((job / f"submission-internal-{attempt}.json").read_text())
    assert internal["answer_input"]["layout"] == "collected"
    assert internal["summary"]["statuses"] == {
        "matched": 1,
        "missing_answer": 60,
        "skipped": 9,
    }
    assert internal["summary"]["match_rate"] == pytest.approx(1 / 61)
    assert json.loads(official.read_text()) == {"existing": True}
    with pytest.raises(ValueError, match="already exists"):
        dispatch(config)


def test_unknown_backend(tmp_path):
    from ddsr_bench.benchmarks.critpt.submission import submit

    with pytest.raises(ValueError, match="unknown submission backend"):
        submit(tmp_path, {"attempts": [0], "backend": "typo"})


def test_reference_path_required(tmp_path):
    with pytest.raises(TypeError, match="references"):
        submission.submit_internal(tmp_path, [0])


@pytest.mark.parametrize("path", [None, "", " "])
def test_empty_reference_path(tmp_path, path):
    with pytest.raises(ValueError, match="explicit references path"):
        submission.submit_internal(tmp_path, [0], references=path)


def test_no_legacy_execution():
    from ddsr_bench.benchmarks.critpt.evaluation.consensus.execution.runtime import (
        Runtime,
    )

    with pytest.raises(ValueError, match="requires the docker backend"):
        Runtime(backend="linux")
    with pytest.raises(TypeError, match="trusted_local"):
        Runtime(trusted_local=True)


def test_trial_results_required(tmp_path, monkeypatch):
    artifacts = tmp_path / "Challenge_1_main__attempt-0" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "answer.py").write_text("def answer(): return 1")
    monkeypatch.setattr(
        submission, "Runtime", lambda **_: pytest.fail("must not start execution")
    )
    with pytest.raises(ValueError, match="no trial results"):
        submitter.submit(tmp_path, {"backend": "internal", "attempts": 0})


@pytest.mark.parametrize("content", ["broken JSON", '{"batches": []}'])
def test_invalid_summary(tmp_path, sample_bundle_path, monkeypatch, content):
    summary = tmp_path / "summary.json"
    summary.write_text(content)
    monkeypatch.setattr(
        submitter, "collect_trials", lambda _: pytest.fail("must not replace summary")
    )
    monkeypatch.setattr(
        submission, "Runtime", lambda **_: pytest.fail("must not start execution")
    )
    with pytest.raises(ValueError):
        submitter.submit(
            tmp_path,
            {
                "backend": "internal",
                "attempts": 0,
                "internal": {"references": str(sample_bundle_path)},
            },
        )
    assert summary.read_text() == content


def test_collected_names(tmp_path):
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
        submission._artifact(tmp_path, load_batch(tmp_path, 2)[0])
        == trial / "solution.py"
    )
    with pytest.raises(ValueError, match="no unique attempt"):
        load_batch(tmp_path, 0)
    rows[0]["attempt"] = 0
    summary.write_text(json.dumps({"batches": [{"attempt": 2, "trials": rows}]}))
    with pytest.raises(ValueError, match="mix attempt"):
        load_batch(tmp_path, 2)


def test_artifact_escape(tmp_path):
    trial = {"problem_id": "Challenge_1_main", "artifact": "../outside.py"}
    with pytest.raises(ValueError, match="inside its job"):
        submission._artifact(tmp_path, trial)


def test_failed_generation(tmp_path, sample_bundle_path, monkeypatch):
    trial = tmp_path / "failed"
    trial.mkdir()
    failure = {"status": "error", "error": "ValueError", "message": "invalid answer"}
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "critpt/Challenge_1_main",
                "trial_name": trial.name,
                "attempt": 0,
                "agent_info": {"name": "teacher", "model_info": {"name": "model"}},
                "static_result": failure,
            }
        )
    )
    monkeypatch.setattr(submission, "Runtime", lambda **_: None)
    monkeypatch.setattr(submission, "provenance", lambda *_: {})

    collect_trials(tmp_path)
    report = submission.submit_internal(
        tmp_path, [0], references=str(sample_bundle_path)
    )

    result = next(r for r in report["results"] if r["problem_id"] == "Challenge_1_main")
    assert result["status"] == "answer_error"
    assert result["error"]["upstream"] == failure


def test_concurrent_grading(tmp_path, sample_bundle_path, monkeypatch):
    (tmp_path / "summary.json").write_text(
        json.dumps({"batches": [{"attempt": 0, "trials": []}]})
    )
    barrier = Barrier(2, timeout=5)
    threads = set()
    grade = submission.grade

    def concurrent(*args, **kwargs):
        threads.add(get_ident())
        barrier.wait()  # Serial execution would fail instead of silently passing.
        return grade(*args, **kwargs)

    monkeypatch.setattr(submission, "grade", concurrent)
    monkeypatch.setattr(submission, "Runtime", lambda **_: None)
    monkeypatch.setattr(submission, "provenance", lambda *_: {})
    report = submission.submit_internal(
        tmp_path, [0], references=str(sample_bundle_path), jobs=2
    )
    assert len(threads) == 2
    assert len(report["results"]) == 70


@pytest.mark.parametrize("outcome", ["different", "unknown", "error"])
def test_multiple_attempts(tmp_path, sample_bundle_path, monkeypatch, outcome):
    from ddsr_bench.benchmarks.critpt.submission import submit

    batches = []
    for attempt in range(2):
        path = tmp_path / f"answer-{attempt}.py"
        path.write_text(f"def answer(): return {attempt + 1}")
        batches.append(
            {
                "attempt": attempt,
                "trials": [
                    {
                        "problem_id": "Challenge_1_main",
                        "attempt": attempt,
                        "artifact": path.name,
                    }
                ],
            }
        )
    (tmp_path / "summary.json").write_text(json.dumps({"batches": batches}))

    def run(payload):
        if payload["action"] == "evaluate":
            value = 2 if "return 2" in payload["code"] else 1
            if value == 2 and outcome == "error":
                return {"status": "error", "stage": "execution"}
            return {"status": "ok", "outputs": [value], "runtime": {}}
        status = "matched" if payload["answer"] == [1] else outcome
        return {
            "status": "ok",
            "comparisons": [
                {
                    "reference": payload["references"][0]["id"],
                    "status": status,
                    "methods": [],
                }
            ],
        }

    runtime = Mock(return_value=SimpleNamespace(run=Mock(side_effect=run)))
    monkeypatch.setattr(submission, "Runtime", runtime)
    monkeypatch.setattr(submission, "provenance", lambda *_: {})
    config = {
        "backend": "internal",
        "attempts": [1, 0],
        "internal": {
            "references": str(sample_bundle_path),
            "jobs": 2,
        },
    }
    submit(tmp_path, config)
    report = json.loads((tmp_path / "submission-internal-0-1.json").read_text())
    assert runtime.call_count == 1
    assert report["reference_cache"] == {"hits": 1, "executions": 1}
    summary = report["summary"]
    assert summary["active"] == 122 and summary["skipped"] == 18
    assert summary["match_rate"] == pytest.approx(1 / 122)
    assert summary["weighted_match_rate"] == pytest.approx(
        sum(item["weighted_match_rate"] for item in summary["attempts"]) / 2
    )
    assert [item["matched"] for item in summary["attempts"]] == [1, 0]
    assert summary["problems"][0]["mean_match_rate"] == 0.5
    assert summary["problems"][5]["mean_match_rate"] is None
    assert summary["statuses"]["missing_answer"] == 120
    assert [r["attempt"] for r in report["results"]] == [0] * 70 + [1] * 70
    # References run once; answers and comparisons are never reused.
    assert runtime.return_value.run.call_count == (4 if outcome == "error" else 5)
    with pytest.raises(ValueError, match="already exists"):
        submit(tmp_path, config | {"attempts": [0, 1]})
    monkeypatch.setattr(submission, "Runtime", lambda **_: pytest.fail("preflight"))
    with pytest.raises(ValueError, match="no unique attempt"):
        submit(tmp_path, config | {"attempts": [0, 2]})
    with pytest.raises(ValueError, match="duplicate problem"):
        submission.summarize(report["results"] + report["results"][:1])
