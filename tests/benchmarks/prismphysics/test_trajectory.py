import json
from dataclasses import asdict
from pathlib import Path

from ddsr_bench.benchmarks.prismphysics.data.loader import load_problem
from ddsr_bench.benchmarks.prismphysics.generation.prompts import messages
from ddsr_bench.training.export import export_sft, export_trajectories, load_trajectory

FIXTURE = Path(__file__).parents[2] / "fixtures" / "prismphysics_02_3006.json"


def test_trajectory(tmp_path: Path) -> None:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    source = load_problem(record, "02")
    trial = tmp_path / "job" / "02-3006__attempt-0"
    (trial / "agent").mkdir(parents=True)
    (trial / "validation").mkdir()
    completion = r"Derivation. $$C=\varepsilon S/d$$"
    public = {
        field: value
        for field, value in asdict(source.spec).items()
        if field in ("raw_id", "shard", "domain", "source", "images")
    }
    response = {
        "problem_id": source.spec.id,
        "problem": public | {"grading_standard": "SECRET_DAG"},
        "strategy": "text-only",
        "model": "teacher",
        "sampling": {"max_tokens": 100},
        "messages": [
            *messages(source.spec),
            {"role": "assistant", "content": completion},
        ],
        "responses": [
            {
                "content": completion,
                "reasoning": "teacher reasoning",
                "finish_reason": "stop",
                "raw": {"ignored": True},
            }
        ],
    }
    (trial / "agent" / "response.json").write_text(json.dumps(response))
    result = {
        "task_name": "prismphysics/02-3006",
        "trial_name": trial.name,
        "config": {"agent": {"kwargs": {"client_name": "vllm"}}},
    }
    (trial / "result.json").write_text(json.dumps(result))
    validation = {
        "reward": 0.5,
        "mode": "prismphysics",
        "status": "different",
        "success": True,
        "threshold": 0.8,
        "direct_matches": [1],
        "credited": [1],
    }
    (trial / "validation" / "result.json").write_text(json.dumps(validation))

    trajectory = load_trajectory(trial)

    assert trajectory.benchmark == "prismphysics"
    assert trajectory.generations[0].completion["reasoning"] == "teacher reasoning"
    assert trajectory.quality["reward"] == 0.5
    assert trajectory.quality["verified"] is False
    assert trajectory.metadata["domain"] == "electromagnetism"
    assert "SECRET_DAG" not in repr(trajectory)
    assert "ignored" not in repr(trajectory)

    trajectories = export_trajectories(trial.parent, tmp_path / "dataset")
    sft = export_sft(trajectories, tmp_path / "dataset")
    sample = json.loads(sft.read_text())
    assert sample["metadata"]["stage"] == "answer"
    assert sample["completion"] == [{"role": "assistant", "content": completion}]
