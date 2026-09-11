import json
from pathlib import Path

from ddsr_bench.benchmarks.cmphysbench.result import trial_fields


def test_harbor_fields(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "response.json").write_text(
        json.dumps(
            {
                "problem": {
                    "answer_type": "Expression",
                    "topic": "Theoretical Foundations",
                }
            }
        )
    )

    fields, artifact = trial_fields({}, tmp_path)

    assert fields == {
        "answer_type": "Expression",
        "topic": "Theoretical Foundations",
    }
    assert artifact == tmp_path / "artifacts" / "answer.txt"
