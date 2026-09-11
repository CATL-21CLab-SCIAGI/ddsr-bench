import json
from pathlib import Path

from ddsr_bench.training.export import export_sft, export_trajectories, load_trajectory


def test_trajectory(tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = job / "7__attempt-0"
    (trial / "agent").mkdir(parents=True)
    (trial / "validation").mkdir()
    response = {
        "problem_id": "7",
        "problem": {
            "context": "context ",
            "question": "question",
            "symbols": "x: value",
            "answer_type": "Expression",
            "topic": "Foundations",
            "reference": "SECRET_REFERENCE",
        },
        "strategy": "single-turn",
        "model": "teacher",
        "sampling": {"max_tokens": 100},
        "messages": [
            {"role": "system", "content": "solve", "sha256": "hash"},
            {"role": "user", "content": "problem", "sha256": "hash"},
            {"role": "assistant", "content": r"work \boxed{x}", "sha256": "hash"},
        ],
        "responses": [
            {
                "content": r"work \boxed{x}",
                "reasoning": "teacher reasoning",
                "finish_reason": "stop",
                "raw": {"ignored": True},
            }
        ],
    }
    (trial / "agent" / "response.json").write_text(json.dumps(response))
    result = {
        "task_name": "cmphysbench/7",
        "trial_name": trial.name,
        "config": {"agent": {"kwargs": {"client_name": "vllm"}}},
    }
    (trial / "result.json").write_text(json.dumps(result))
    validation = {
        "reward": 0.75,
        "seed_score": 75,
        "mode": "seed",
        "status": "different",
    }
    (trial / "validation" / "result.json").write_text(json.dumps(validation))

    trajectory = load_trajectory(trial)

    assert trajectory.benchmark == "cmphysbench"
    assert trajectory.generations[0].completion["reasoning"] == "teacher reasoning"
    assert trajectory.quality["seed_score"] == 75
    assert "ignored" not in repr(trajectory)
    assert "SECRET_REFERENCE" not in repr(trajectory)
    trajectories = export_trajectories(job, tmp_path / "dataset")
    sft = export_sft(trajectories, tmp_path / "dataset")
    sample = json.loads(sft.read_text())
    assert sample["metadata"]["stage"] == "answer"
    assert "reasoning" not in sample["completion"][0]
