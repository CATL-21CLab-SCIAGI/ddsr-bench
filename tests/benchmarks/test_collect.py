import csv
import json
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.collect import collect_trials


def write_trial(
    job: Path,
    name: str,
    problem_id: str,
    started_at: str,
    reward: float,
    *,
    agent: str = "critpt",
) -> None:
    trial = job / name
    (trial / "artifacts").mkdir(parents=True)
    (trial / "verifier").mkdir()
    (trial / "artifacts" / "answer.py").write_text("def answer():\n    pass\n")
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": f"critpt/{problem_id}",
                "trial_name": name,
                "started_at": started_at,
                "agent_info": {
                    "name": agent,
                    "model_info": {"name": "model", "provider": "vllm"},
                },
                "config": {"agent": {"kwargs": {"style": "one-step"}}},
                "verifier_result": {"rewards": {"reward": reward}},
            }
        )
    )
    (trial / "verifier" / "result.json").write_text(
        json.dumps({"status": "passed" if reward else "different"})
    )


def test_collects_complete_batches(tmp_path: Path) -> None:
    write_trial(tmp_path, "a-late", "a", "2026-01-02", 0)
    write_trial(tmp_path, "a-early", "a", "2026-01-01", 1)
    write_trial(tmp_path, "a-extra", "a", "2026-01-03", 1)
    write_trial(tmp_path, "b-late", "b", "2026-01-02", 1)
    write_trial(tmp_path, "b-early", "b", "2026-01-01", 0)

    summary = collect_trials(tmp_path)

    assert summary["complete_batches"] == 2
    assert summary["unbatched_trials"] == 1
    assert [row["trial_name"] for row in summary["batches"][0]["trials"]] == [
        "a-early",
        "b-early",
    ]
    assert summary["batches"][0]["passed"] == 1

    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved == summary
    with (tmp_path / "summary.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert rows[0]["attempt"] == "0"
    assert rows[0]["answer"] == "a-early/artifacts/answer.py"


def test_rejects_non_critpt_trial(tmp_path: Path) -> None:
    write_trial(tmp_path, "trial", "problem", "2026-01-01", 1)
    path = tmp_path / "trial" / "result.json"
    data = json.loads(path.read_text())
    data["task_name"] = "other/problem"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="not a CritPt trial"):
        collect_trials(tmp_path)


def test_rejects_multiple_agents(tmp_path: Path) -> None:
    write_trial(tmp_path, "first", "a", "2026-01-01", 1)
    write_trial(tmp_path, "second", "b", "2026-01-01", 1, agent="other")

    with pytest.raises(ValueError, match="multiple agent"):
        collect_trials(tmp_path)
