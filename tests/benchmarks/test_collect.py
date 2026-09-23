import csv
import json
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.collect import collect_trials, group_results, load_batch


def test_group_results():
    trials = [
        {"attempt": 2, "benchmark_config": {"tag": "z"}, "reward": 0.25},
        {"attempt": 0, "benchmark_config": {"tag": "a"}, "reward": 1.0},
        {"attempt": 2, "benchmark_config": {"tag": "a"}, "reward": 0.5},
    ]

    def metrics(rows):
        return {"total": sum(row["reward"] for row in rows)}

    by_tag = group_results(trials, "tag", metrics)
    assert list(by_tag) == ["a", "z"]
    assert by_tag == {"a": {"total": 1.5}, "z": {"total": 0.25}}
    assert group_results(trials, "attempt", metrics) == {
        "0": {"total": 1.0},
        "2": {"total": 0.75},
    }
    assert group_results([], "tag", metrics) == {}


def write_trial(
    job: Path,
    name: str,
    problem_id: str,
    started_at: str,
    reward: float,
    *,
    agent: str = "critpt",
    benchmark: str = "critpt",
) -> None:
    trial = job / name
    (trial / "artifacts").mkdir(parents=True)
    (trial / "verifier").mkdir()
    artifacts = {
        "critpt": "answer.py",
        "scicode": "solution.py",
        "cmphysbench": "answer.txt",
        "phybench": "answer.txt",
    }
    artifact = artifacts[benchmark]
    (trial / "artifacts" / artifact).write_text("def answer():\n    pass\n")
    kwargs = (
        {"style": "one-step"} if benchmark == "critpt" else {"with_background": False}
    )
    result = {
        "task_name": f"{benchmark}/{problem_id}",
        "trial_name": name,
        "started_at": started_at,
        "agent_info": {
            "name": agent,
            "model_info": {"name": "model", "provider": "vllm"},
        },
        "config": {"agent": {"kwargs": kwargs}},
        "verifier_result": {"rewards": {"reward": reward}},
    }
    if benchmark in ("cmphysbench", "phybench"):
        static_result = {
            "reward": reward,
            "status": "passed" if reward == 1 else "different",
        }
        if benchmark == "cmphysbench":
            static_result.update(
                {
                    "answer_type": "Expression",
                    "topic": "Theoretical Foundations",
                }
            )
        else:
            static_result["tag"] = "MECHANICS"
        result["static_result"] = static_result
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "verifier" / "result.json").write_text(
        json.dumps({"status": "passed" if reward else "different"})
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_collects_all_batches(tmp_path: Path, explicit: bool) -> None:
    write_trial(tmp_path, "a-late", "a", "2026-01-02", 0)
    write_trial(tmp_path, "a-early", "a", "2026-01-01", 1)
    write_trial(tmp_path, "a-extra", "a", "2026-01-03", 1)
    write_trial(tmp_path, "b-late", "b", "2026-01-02", 1)
    write_trial(tmp_path, "b-early", "b", "2026-01-01", 0)
    if explicit:
        for name, attempt in [
            ("a-early", 0),
            ("a-late", 1),
            ("a-extra", 2),
            ("b-early", 0),
            ("b-late", 1),
        ]:
            path = tmp_path / name / "result.json"
            path.write_text(
                json.dumps(json.loads(path.read_text()) | {"attempt": attempt})
            )

    summary = collect_trials(tmp_path)

    assert summary["complete_batches"] == 2
    assert summary["incomplete_batches"] == 1
    assert summary["unbatched_trials"] == 0
    assert summary["batches"][2]["complete"] is False
    assert [trial["trial_name"] for trial in load_batch(tmp_path, 2)] == ["a-extra"]
    assert [row["trial_name"] for row in summary["batches"][0]["trials"]] == [
        "a-early",
        "b-early",
    ]
    assert summary["batches"][0]["passed"] == 1

    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved == summary
    with (tmp_path / "summary.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 5
    assert rows[0]["attempt"] == "0"
    assert rows[0]["artifact"] == "a-early/artifacts/answer.py"


def test_collects_scicode(tmp_path: Path) -> None:
    write_trial(
        tmp_path,
        "trial",
        "19",
        "2026-01-01",
        1,
        agent="scicode",
        benchmark="scicode",
    )

    summary = collect_trials(tmp_path)
    trial = summary["batches"][0]["trials"][0]

    assert trial["benchmark"] == "scicode"
    assert trial["benchmark_config"] == {"with_background": False}
    assert trial["artifact"] == "trial/artifacts/solution.py"


def test_collects_cmphysbench(tmp_path: Path) -> None:
    write_trial(
        tmp_path,
        "trial",
        "7",
        "2026-01-01",
        0.75,
        agent="cmphysbench",
        benchmark="cmphysbench",
    )

    summary = collect_trials(tmp_path)
    trial = summary["batches"][0]["trials"][0]

    assert trial["benchmark_config"] == {
        "answer_type": "Expression",
        "topic": "Theoretical Foundations",
    }
    assert trial["artifact"] == "trial/artifacts/answer.txt"
    assert summary["metrics"]["overall"] == {
        "trials": 1,
        "mean_seed": 75.0,
        "accuracy": 0.0,
    }
    assert (
        summary["metrics"]["by_topic"]["Theoretical Foundations"]
        == summary["metrics"]["overall"]
    )
    assert summary["metrics"]["by_answer_type"]["Expression"]["mean_seed"] == 75
    assert summary["metrics"]["by_attempt"]["0"]["accuracy"] == 0


def test_collects_phybench(tmp_path: Path) -> None:
    write_trial(
        tmp_path,
        "trial",
        "133",
        "2026-01-01",
        0.8,
        agent="phybench",
        benchmark="phybench",
    )

    summary = collect_trials(tmp_path)
    trial = summary["batches"][0]["trials"][0]

    assert trial["benchmark_config"] == {"tag": "MECHANICS"}
    assert trial["artifact"] == "trial/artifacts/answer.txt"
    assert summary["metrics"]["overall"] == {
        "trials": 1,
        "mean_eed": 80.0,
        "accuracy": 0.0,
    }
    assert summary["metrics"]["by_tag"]["MECHANICS"]["mean_eed"] == 80


def test_rejects_unknown_benchmark(tmp_path: Path) -> None:
    write_trial(tmp_path, "trial", "problem", "2026-01-01", 1)
    path = tmp_path / "trial" / "result.json"
    data = json.loads(path.read_text())
    data["task_name"] = "other/problem"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="unsupported benchmark"):
        collect_trials(tmp_path)


def test_rejects_multiple_agents(tmp_path: Path) -> None:
    write_trial(tmp_path, "first", "a", "2026-01-01", 1)
    write_trial(tmp_path, "second", "b", "2026-01-01", 1, agent="other")

    with pytest.raises(ValueError, match="multiple benchmarks, agents"):
        collect_trials(tmp_path)


@pytest.mark.parametrize("source", ["static", "verifier", "exception"])
def test_failure_details(tmp_path, source):
    write_trial(tmp_path, "trial", "Challenge_1_main", "2026-01-01", 0)
    path = tmp_path / "trial" / "result.json"
    verifier = path.parent / "verifier" / "result.json"
    data = json.loads(path.read_text())
    failure = {"status": "error", "error": "ValueError", "message": "invalid answer"}
    if source == "static":
        data["static_result"] = failure
    elif source == "verifier":
        verifier.write_text(json.dumps(failure))
    else:
        failure = {"exception_type": "RuntimeError", "exception_message": "failed"}
        data["exception_info"] = failure
        verifier.unlink()
    path.write_text(json.dumps(data))

    summary = collect_trials(tmp_path)

    trial = summary["batches"][0]["trials"][0]
    assert trial["status"] == "error"
    assert trial["error"] == failure
    with (tmp_path / "summary.csv").open(newline="") as file:
        assert json.loads(next(csv.DictReader(file))["error"]) == failure
