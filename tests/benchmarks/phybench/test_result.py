import json
from pathlib import Path

from ddsr_bench.benchmarks.phybench.result import summarize, trial_fields


def test_harbor_fields(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "response.json").write_text(json.dumps({"problem": {"tag": "MECHANICS"}}))

    fields, artifact = trial_fields({}, tmp_path)

    assert fields == {"tag": "MECHANICS"}
    assert artifact == tmp_path / "artifacts" / "answer.txt"


def test_summary() -> None:
    trials = [
        {"reward": 1.0, "attempt": 0, "benchmark_config": {"tag": "MECHANICS"}},
        {
            "reward": 0.25,
            "attempt": 0,
            "benchmark_config": {"tag": "ELECTROMAGNETISM"},
        },
        {"reward": None, "attempt": 1, "benchmark_config": {"tag": "MECHANICS"}},
    ]

    metrics = summarize(trials)

    assert metrics["overall"] == {
        "trials": 2,
        "mean_eed": 62.5,
        "accuracy": 0.5,
    }
    assert metrics["by_tag"]["MECHANICS"]["accuracy"] == 1
    assert metrics["by_attempt"]["0"] == metrics["overall"]
