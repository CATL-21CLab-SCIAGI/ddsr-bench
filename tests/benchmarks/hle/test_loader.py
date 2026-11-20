import json
from base64 import b64encode
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.hle.data.loader import load_problem

FIXTURE = Path(__file__).parents[2] / "fixtures" / "hle_gold.json"


def _record() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_real_record_and_public_projection() -> None:
    problem = load_problem(_record())
    public = json.dumps(asdict(problem.spec))

    assert problem.spec.id == "668825f80a642802bdfeadfa"
    assert problem.spec.subset == "Gold"
    assert problem.spec.image is None
    assert problem.answer is not None
    assert problem.answer.answer == "D"
    assert problem.answer.type == "multipleChoice"
    assert problem.answer.rationale_image is None
    assert '"answer"' not in public
    assert "rationale" not in public


def test_image_and_mismatched_flattening() -> None:
    record = _record()
    payload = json.loads(record["json"])
    payload["image"] = "data:image/png;base64,AA=="
    record["json"] = json.dumps(payload)
    assert load_problem(record).spec.image == "data:image/png;base64,AA=="

    record["question"] = "changed"
    with pytest.raises(ValueError, match="flattened fields"):
        load_problem(record)


def test_rationale_image_stays_verifier_side() -> None:
    record = _record()
    payload = json.loads(record["json"])
    encoded = b64encode(b"\x89PNG\r\n\x1a\ncontent").decode()
    payload["rationale_image"] = {
        "bytes": {"__base64__": encoded},
        "path": None,
    }
    record["json"] = json.dumps(payload)

    problem = load_problem(record)

    assert problem.answer is not None
    assert problem.answer.rationale_image == f"data:image/png;base64,{encoded}"
    assert "rationale_image" not in asdict(problem.spec)
