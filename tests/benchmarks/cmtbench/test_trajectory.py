import json
from pathlib import Path

from ddsr_bench.benchmarks.cmtbench.data.loader import load_problem
from ddsr_bench.benchmarks.cmtbench.generation.prompts import messages
from ddsr_bench.training.export import export_sft, export_trajectories, load_trajectory

FIXTURE = Path(__file__).parents[2] / "fixtures" / "cmtbench_39.json"


def test_trajectory(tmp_path: Path) -> None:
    source = load_problem(json.loads(FIXTURE.read_text(encoding="utf-8")))
    trial = tmp_path / "job" / "39__attempt-0"
    (trial / "agent").mkdir(parents=True)
    (trial / "validation").mkdir()
    completion = r"Derivation. \boxed{x+x}"
    response = {
        "problem_id": source.spec.id,
        "problem": {
            "prompt": source.spec.prompt,
            "type": source.spec.type,
            "solution": "SECRET_SOLUTION",
        },
        "strategy": "single-turn",
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
        "task_name": "cmtbench/39",
        "trial_name": trial.name,
        "config": {"agent": {"kwargs": {"client_name": "vllm"}}},
    }
    (trial / "result.json").write_text(json.dumps(result))
    validation = {
        "reward": 1.0,
        "mode": "cmtbench",
        "status": "passed",
        "method": "parameterized",
        "success": True,
    }
    (trial / "validation" / "result.json").write_text(json.dumps(validation))

    trajectory = load_trajectory(trial)

    assert trajectory.benchmark == "cmtbench"
    assert trajectory.generations[0].completion["reasoning"] == "teacher reasoning"
    assert trajectory.quality["method"] == "parameterized"
    assert trajectory.quality["parsed"] is True
    assert trajectory.metadata == {
        "prompt": source.spec.prompt,
        "type": "Other",
    }
    assert "SECRET_SOLUTION" not in repr(trajectory)
    assert "ignored" not in repr(trajectory)

    trajectories = export_trajectories(trial.parent, tmp_path / "dataset")
    sft = export_sft(trajectories, tmp_path / "dataset")
    sample = json.loads(sft.read_text())
    assert sample["metadata"]["stage"] == "answer"
    assert sample["completion"] == [{"role": "assistant", "content": completion}]
