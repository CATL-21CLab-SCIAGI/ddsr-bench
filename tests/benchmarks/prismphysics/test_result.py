import json
from pathlib import Path

from ddsr_bench.benchmarks.prismphysics.result import summarize, trial_fields


def test_harbor_fields(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "response.json").write_text(
        json.dumps({"problem": {"domain": "optics", "shard": "03"}})
    )

    fields, artifact = trial_fields({}, tmp_path)

    assert fields == {"domain": "optics", "shard": "03"}
    assert artifact == tmp_path / "artifacts" / "answer.txt"


def test_summary() -> None:
    trials = [
        {
            "reward": 1.0,
            "status": "passed",
            "attempt": 0,
            "benchmark_config": {"domain": "optics", "shard": "03"},
        },
        {
            "reward": 0.5,
            "status": "different",
            "attempt": 0,
            "benchmark_config": {"domain": "mechanics", "shard": "01"},
        },
        {
            "reward": None,
            "status": "missing",
            "attempt": 1,
            "benchmark_config": {"domain": "optics", "shard": "03"},
        },
    ]

    metrics = summarize(trials)

    assert metrics["overall"] == {
        "trials": 2,
        "step_score": 75.0,
        "accuracy": 50.0,
        "parse_rate": 100.0,
    }
    assert metrics["by_domain"]["optics"]["accuracy"] == 100
    assert metrics["by_attempt"]["0"] == metrics["overall"]
