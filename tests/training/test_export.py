import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.critpt.generation.prompts import system_prompt
from ddsr_bench.benchmarks.critpt.trajectory import sft_samples as critpt_samples
from ddsr_bench.training.export import (
    export_sft,
    export_trajectories,
    load_trajectory,
)
from ddsr_bench.training.schemas import Generation, Trajectory


def write_trial(job: Path, name: str = "Challenge_2_sub_1__attempt-0") -> Path:
    trial = job / name
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    response = {
        "problem_id": "Challenge_2_sub_1",
        "problem": {
            "statement": "problem",
            "code_template": "def answer():\n    return ...",
        },
        "strategy": "two-step",
        "model": "teacher",
        "sampling": {"model": "teacher", "seed": 7},
        "messages": [
            {"role": "system", "content": "solve"},
            {"role": "user", "content": "problem"},
            {"role": "assistant", "content": "derivation"},
            {"role": "user", "content": "format"},
            {"role": "assistant", "content": "```python\ncode\n```"},
        ],
        "responses": [
            {
                "content": "derivation",
                "reasoning": "private thought",
                "finish_reason": "stop",
                "raw": {},
            },
            {
                "content": "```python\ncode\n```",
                "reasoning": None,
                "finish_reason": "stop",
                "raw": {},
            },
        ],
    }
    path = trial / "agent" / "response.json"
    path.write_text(json.dumps(response), encoding="utf-8")
    result = {
        "task_name": "critpt/Challenge_2_sub_1",
        "trial_name": name,
        "config": {"agent": {"kwargs": {"client_name": "vllm"}}},
    }
    (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
    verification = {"reward": 1.0, "mode": "reference", "status": "passed"}
    (trial / "verifier" / "result.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    return path


def test_normalization(tmp_path: Path) -> None:
    job = tmp_path / "job"
    trajectory = load_trajectory(write_trial(job).parent.parent)

    assert trajectory.schema_version == 2
    assert trajectory.benchmark == "critpt"
    assert [generation.id for generation in trajectory.generations] == [
        "derivation",
        "formatting",
    ]
    assert trajectory.metadata["challenge_id"] == "Challenge_2"
    assert trajectory.metadata["problem_type"] == "sub"
    assert trajectory.metadata["problem_index"] == 1
    assert trajectory.metadata["statement"] == "problem"
    assert trajectory.generations[0].completion["reasoning"] == "private thought"
    assert trajectory.teacher["client"] == "vllm"
    assert trajectory.quality["verified"] is True
    assert "challenge_id" not in asdict(trajectory)
    assert "raw" not in repr(trajectory)


def test_export(tmp_path: Path) -> None:
    job = tmp_path / "job"
    write_trial(job, "Challenge_2_sub_1__attempt-1")
    write_trial(job, "Challenge_2_sub_1__attempt-0")

    output = export_trajectories(job, tmp_path / "dataset")
    records = [json.loads(line) for line in output.read_text().splitlines()]

    assert [record["id"] for record in records] == [
        "Challenge_2_sub_1__attempt-0",
        "Challenge_2_sub_1__attempt-1",
    ]


def test_rejects_mixed_benchmarks(tmp_path: Path) -> None:
    job = tmp_path / "job"
    write_trial(job)
    scicode = job / "scicode-19"
    (scicode / "agent").mkdir(parents=True)
    (scicode / "agent" / "response.json").write_text("{}")
    (scicode / "result.json").write_text(
        json.dumps({"task_name": "scicode/19"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="multiple benchmarks"):
        export_trajectories(job, tmp_path / "dataset")


def test_sft_views(tmp_path: Path) -> None:
    job = tmp_path / "job"
    response = write_trial(job)
    dataset = tmp_path / "dataset"
    trajectories = export_trajectories(job, dataset)

    output = export_sft(trajectories, dataset)
    samples = [json.loads(line) for line in output.read_text().splitlines()]

    assert [sample["metadata"]["stage"] for sample in samples] == [
        "derivation",
        "formatting",
        "answer",
    ]
    assert [len(sample["prompt"]) for sample in samples] == [2, 4, 2]
    assert all(len(sample["completion"]) == 1 for sample in samples)
    assert "reasoning" not in samples[0]["completion"][0]

    derivation = export_sft(trajectories, tmp_path / "derivation", view="derivation")
    assert len(derivation.read_text().splitlines()) == 1

    native = export_sft(trajectories, tmp_path / "native", view="native")
    native_samples = [json.loads(line) for line in native.read_text().splitlines()]
    assert [sample["metadata"]["stage"] for sample in native_samples] == [
        "derivation",
        "formatting",
    ]

    answer = critpt_samples(load_trajectory(response.parent.parent), "answer")[0]
    assert answer.prompt[0]["content"] == system_prompt("one-step")
    assert answer.prompt[1]["content"].endswith(
        "```python\ndef answer():\n    return ...\n```"
    )
    assert all(message["content"] != "derivation" for message in answer.prompt)
    assert answer.completion[0]["content"] == ("derivation\n\n```python\ncode\n```")
    assert answer.metadata["derived"] is True

    formatting = critpt_samples(load_trajectory(response.parent.parent), "formatting")
    assert len(formatting) == 1
    assert len(formatting[0].prompt) == 4


def test_one_step_views() -> None:
    trajectory = Trajectory(
        schema_version=2,
        id="trial",
        benchmark="critpt",
        problem_id="problem",
        generations=(
            Generation(
                "answer",
                (
                    {"role": "system", "content": "solve"},
                    {"role": "user", "content": "problem and template"},
                ),
                {"role": "assistant", "content": "reasoned answer"},
                "model",
                {},
            ),
        ),
        teacher={"strategy": "one-step"},
        quality={},
        metadata={
            "challenge_id": "challenge",
            "problem_type": "main",
            "problem_index": None,
            "statement": "problem",
            "code_template": "def answer():\n    return ...",
        },
        provenance={},
    )

    assert critpt_samples(trajectory, "full")[0].metadata["stage"] == "answer"
    assert len(critpt_samples(trajectory, "native")) == 1
    assert critpt_samples(trajectory, "full")[0].metadata["derived"] is False
    assert len(critpt_samples(trajectory, "answer")) == 1
    with pytest.raises(ValueError, match="unavailable"):
        critpt_samples(trajectory, "derivation")
    with pytest.raises(ValueError, match="unavailable"):
        critpt_samples(trajectory, "formatting")
    with pytest.raises(ValueError, match="requires 2.*found 1"):
        critpt_samples(replace(trajectory, teacher={"strategy": "two-step"}), "full")
