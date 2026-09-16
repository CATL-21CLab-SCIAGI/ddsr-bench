import json
from pathlib import Path

from ddsr_bench.benchmarks.phybench.data.loader import load_problem
from ddsr_bench.benchmarks.phybench.generation.prompts import messages
from ddsr_bench.training.export import export_sft, export_trajectories, load_trajectory

FIXTURE = Path(__file__).parents[2] / "fixtures" / "phybench_133.json"


def test_trajectory(tmp_path: Path) -> None:
    source = load_problem(json.loads(FIXTURE.read_text(encoding="utf-8")))
    trial = tmp_path / "job" / "133__attempt-0"
    (trial / "agent").mkdir(parents=True)
    (trial / "validation").mkdir()
    completion = r"Derivation. \boxed{\frac{v^3}{R v_x}}"
    prompt = messages(source.spec)[0]
    response = {
        "problem_id": source.spec.id,
        "problem": {"tag": source.spec.tag, "content": source.spec.content},
        "strategy": "single-turn",
        "model": "teacher",
        "sampling": {"max_tokens": 100},
        "messages": [prompt, {"role": "assistant", "content": completion}],
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
        "task_name": "phybench/133",
        "trial_name": trial.name,
        "config": {"agent": {"kwargs": {"client_name": "vllm"}}},
    }
    (trial / "result.json").write_text(json.dumps(result))
    validation = {
        "reward": 1.0,
        "eed_score": 100,
        "mode": "eed",
        "status": "passed",
    }
    (trial / "validation" / "result.json").write_text(json.dumps(validation))

    trajectory = load_trajectory(trial)

    assert trajectory.benchmark == "phybench"
    assert trajectory.generations[0].completion["reasoning"] == "teacher reasoning"
    assert trajectory.metadata == {
        "tag": "MECHANICS",
        "content": source.spec.content,
    }
    assert trajectory.quality["eed_score"] == 100
    assert "natural coordinate system" not in repr(trajectory)
    assert "ignored" not in repr(trajectory)

    trajectories = export_trajectories(trial.parent, tmp_path / "dataset")
    sft = export_sft(trajectories, tmp_path / "dataset")
    sample = json.loads(sft.read_text())
    assert sample["metadata"]["stage"] == "answer"
    assert sample["completion"] == [{"role": "assistant", "content": completion}]
