import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.prismphysics.data.loader import (
    DATASET,
    FILES,
    REVISION,
    THRESHOLD,
    load_problem,
    load_problems,
    load_shard,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "prismphysics_02_3006.json"


def record() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_real_problem() -> None:
    problem = load_problem(record(), "02")

    assert problem.spec.id == "02-3006"
    assert problem.spec.domain == "electromagnetism"
    assert problem.spec.subquestions[0].letter == "a"
    assert problem.spec.images[0].caption == "Fig. 3.10"
    assert problem.answer is not None
    assert problem.answer.formulas[1].dependencies == (1,)


def test_public_projection() -> None:
    problem = load_problem(record(), "02")
    public = json.dumps(asdict(problem.spec))

    assert "parallel-plate capacitor" not in public
    assert "grading_standard" not in public
    assert "$$C=" not in public


def test_shard_namespace(tmp_path: Path) -> None:
    path = tmp_path / "02_cleaned_dag.json"
    path.write_text(json.dumps([record()]), encoding="utf-8")

    assert load_shard(path)[0].spec.id == "02-3006"
    assert load_problem(record(), "03").spec.id == "03-3006"


def test_invalid_dag() -> None:
    value = record()
    value["grading_standard"][1]["dependency"] = [99]

    with pytest.raises(ValueError, match="invalid formula DAG"):
        load_problem(value, "02")


def test_cyclic_dag() -> None:
    value = record()
    value["grading_standard"][0]["dependency"] = [2]

    with pytest.raises(ValueError, match="contains a cycle"):
        load_problem(value, "02")


def test_unique_ids() -> None:
    with pytest.raises(ValueError, match="duplicate IDs"):
        load_problems([record(), record()], "02")


def test_rejects_malformed_public_text() -> None:
    value = record()
    value["subquestions"][0]["subproblem"] = {"unexpected": "object"}

    with pytest.raises(TypeError, match="requires text at 'subproblem'"):
        load_problem(value, "02")

    del value["subquestions"][0]["subproblem"]
    assert load_problem(value, "02").spec.subquestions[0].prompt == ""


def test_pinned_source() -> None:
    assert DATASET == "open-prism/PRISM-Physics"
    assert REVISION == "89d1e1cb91ec14b319482ec36b43fca5305f9c86"
    assert FILES == tuple(f"{index:02d}_cleaned_dag.json" for index in range(1, 8))
    assert THRESHOLD == 0.8
