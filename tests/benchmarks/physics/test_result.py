from pathlib import Path

import pytest

from ddsr_bench.benchmarks.collect import collect_trials
from ddsr_bench.benchmarks.physics.result import summarize, trial_fields
from ddsr_bench.benchmarks.utils import write_json


def test_summary() -> None:
    trials = [
        {
            "attempt": 0,
            "status": "passed",
            "reward": 1.0,
            "benchmark_config": {
                "split": "test",
                "domain": "Optics",
                "answer_types": ["numeric"],
                "answer_correct": [True],
                "confidence": 90,
            },
        },
        {
            "attempt": 0,
            "status": "different",
            "reward": 0.0,
            "benchmark_config": {
                "split": "test",
                "domain": "Optics",
                "answer_types": ["symbolic"],
                "answer_correct": [False],
                "confidence": 70,
            },
        },
    ]

    result = summarize(trials)

    assert result["overall"]["strict_accuracy"] == 0.5
    assert result["overall"]["answer_accuracy"] == 0.5
    assert result["by_split"]["test"]["problems"] == 2
    assert result["by_attempt"]["0"]["partial_credit"] == 0.5


def test_collection_adapter(tmp_path: Path) -> None:
    trial = tmp_path / "p1__attempt-0"
    (trial / "artifacts").mkdir(parents=True)
    (trial / "artifacts" / "answer.txt").write_text("answer", encoding="utf-8")
    result = {
        "task_name": "physics/p1",
        "trial_name": trial.name,
        "attempt": 0,
        "started_at": "2026-01-01T00:00:00+00:00",
        "agent_info": {
            "name": "physics",
            "model_info": {"provider": "vllm", "name": "teacher"},
        },
        "static_result": {
            "reward": 1.0,
            "status": "passed",
            "judged": True,
            "split": "test",
            "domain": "Optics",
            "answer_types": ["numeric"],
            "answer_correct": [True],
            "confidence": 90,
            "judge_model": "judge",
        },
    }
    write_json(trial / "result.json", result)

    summary = collect_trials(tmp_path)

    assert summary["metrics"]["overall"]["strict_accuracy"] == 1
    assert summary["metrics"]["overall"]["unjudged"] == 0


def test_rejects_misaligned_verdicts(tmp_path: Path) -> None:
    result = {
        "static_result": {
            "split": "test",
            "domain": "Optics",
            "answer_types": ["numeric"],
            "answer_correct": [True, False],
            "confidence": 90,
            "judge_model": "judge",
        }
    }

    with pytest.raises(TypeError, match="answer verdicts"):
        trial_fields(result, tmp_path)
