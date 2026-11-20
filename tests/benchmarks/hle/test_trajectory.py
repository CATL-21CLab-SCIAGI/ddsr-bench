import json
from pathlib import Path

from ddsr_bench.benchmarks.hle.trajectory import normalize, sft_samples


def test_normalizes_multimodal_generation(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    messages = [
        {"role": "system", "content": "format", "sha256": "a"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "question"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
            "sha256": "b",
        },
        {"role": "assistant", "content": "answer", "sha256": "c"},
    ]
    response = {
        "problem_id": "p1",
        "problem": {
            "subset": "Gold",
            "category": "Physics",
            "subject": "Mechanics",
            "question": "question",
            "image": True,
        },
        "model": "teacher",
        "strategy": "single-turn",
        "sampling": {},
        "messages": messages,
        "responses": [
            {"content": "answer", "reasoning": "thought", "finish_reason": "stop"}
        ],
    }
    (agent / "response.json").write_text(json.dumps(response), encoding="utf-8")
    result = {
        "trial_name": "trial",
        "config": {"agent": {"kwargs": {"client_name": "vllm"}}},
    }
    validation = {
        "mode": "hle",
        "status": "passed",
        "reward": 1.0,
        "judge_model": "judge",
    }

    trajectory = normalize(trial, response, result, validation)
    samples = sft_samples(trajectory, "native")

    assert trajectory.benchmark == "hle"
    assert trajectory.quality["verified"]
    assert samples[0].prompt[1]["content"][1]["type"] == "image_url"
    assert samples[0].completion[0]["content"] == "answer"
