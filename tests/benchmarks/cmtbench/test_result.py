import json
from pathlib import Path

from ddsr_bench.benchmarks.cmtbench.result import summarize, trial_fields


def test_harbor_fields(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "response.json").write_text(json.dumps({"problem": {"type": "DMRG"}}))

    fields, artifact = trial_fields({}, tmp_path)

    assert fields == {"type": "DMRG"}
    assert artifact == tmp_path / "artifacts" / "answer.txt"


def test_summary() -> None:
    trials = [
        {
            "reward": 1.0,
            "status": "passed",
            "attempt": 0,
            "benchmark_config": {"type": "DMRG"},
        },
        {
            "reward": 0.0,
            "status": "error",
            "attempt": 0,
            "benchmark_config": {"type": "QMC"},
        },
        {
            "reward": None,
            "status": "missing",
            "attempt": 1,
            "benchmark_config": {"type": "DMRG"},
        },
    ]

    metrics = summarize(trials)

    assert metrics["overall"] == {
        "trials": 2,
        "pass_rate": 50.0,
        "parse_rate": 50.0,
    }
    assert metrics["by_type"]["DMRG"]["pass_rate"] == 100
    assert metrics["by_attempt"]["0"] == metrics["overall"]
