import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.physics.data.loader import load_problem

FIXTURE = Path(__file__).parents[2] / "fixtures" / "physics_atomic_1_1.json"


def _record() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_real_record() -> None:
    problem = load_problem(_record(), "validation")

    assert problem.spec.id == "atomic/1-1"
    assert problem.spec.domain == "Atomic Physics"
    assert problem.spec.split == "validation"
    assert problem.spec.figures == ()
    assert problem.answer is not None
    assert problem.answer.labels == ("1",)
    assert problem.answer.types == ("text",)


def test_public_projection() -> None:
    problem = load_problem(_record(), "validation")
    public = json.dumps(asdict(problem.spec))

    assert "final_answers" not in public
    assert "solutions" not in public
    assert "electron microscope" in public


def test_figure() -> None:
    record = _record()
    record["figures"] = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AA=="},
        }
    ]

    figure = load_problem(record, "test").spec.figures[0]

    assert figure.url == "data:image/png;base64,AA=="


def test_rejects_misaligned_answers() -> None:
    record = _record()
    record["answer_labels"] = []

    with pytest.raises(TypeError, match="nonempty list"):
        load_problem(record, "validation")
