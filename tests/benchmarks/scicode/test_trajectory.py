import json
from pathlib import Path

from ddsr_bench.training.export import export_sft, export_trajectories, load_trajectory


def write_trial(job: Path) -> Path:
    trial = job / "19__attempt-0"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    response = {
        "problem_id": "19",
        "with_background": False,
        "steps": [
            {
                "id": "19.1",
                "prompt": "Implement tensor().",
                "response": {
                    "content": "```python\ndef tensor(): pass\n```",
                    "reasoning": "teacher reasoning",
                    "model": "teacher",
                    "finish_reason": "stop",
                    "raw": {"secret": True},
                },
                "code": "def tensor(): pass",
                "source": "model",
            },
            {
                "id": "19.2",
                "code": "def n_tangle(): pass",
                "source": "fixed",
            },
        ],
    }
    (trial / "agent" / "response.json").write_text(json.dumps(response))
    result = {
        "task_name": "scicode/19",
        "trial_name": trial.name,
        "config": {
            "agent": {
                "model_name": "teacher",
                "kwargs": {"client_name": "bedrock"},
            }
        },
    }
    (trial / "result.json").write_text(json.dumps(result))
    validation = {
        "reward": 1.0,
        "mode": "scicode",
        "status": "passed",
        "steps": [{"id": "19.1", "status": "passed", "tests": "SECRET_TESTS"}],
    }
    (trial / "verifier" / "result.json").write_text(json.dumps(validation))
    return trial


def test_scicode_trajectory(tmp_path: Path) -> None:
    job = tmp_path / "job"
    trajectory = load_trajectory(write_trial(job))

    assert trajectory.benchmark == "scicode"
    assert trajectory.problem_id == "19"
    assert [(item.id, item.source) for item in trajectory.generations] == [
        ("19.1", "model"),
        ("19.2", "fixed"),
    ]
    assert trajectory.generations[0].prompt[0]["content"] == "Implement tensor()."
    assert trajectory.generations[0].completion["reasoning"] == "teacher reasoning"
    assert trajectory.generations[0].quality == {"status": "passed"}
    assert trajectory.generations[1].prompt == ()
    assert trajectory.teacher["client"] == "bedrock"
    assert trajectory.quality["verified"] is True
    assert "SECRET_TESTS" not in repr(trajectory)
    assert "secret" not in repr(trajectory)
    assert trajectory.provenance["upstream_commit"]

    trajectories = export_trajectories(job, tmp_path / "dataset")
    output = export_sft(trajectories, tmp_path / "dataset")
    samples = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(samples) == 1
    assert samples[0]["metadata"]["stage"] == "19.1"
    assert "reasoning" not in samples[0]["completion"][0]
