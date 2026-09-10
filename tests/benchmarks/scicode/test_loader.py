import json
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.scicode.generation import render_prompt
from ddsr_bench.benchmarks.scicode.loader import load_problem

FIXTURE = Path(__file__).parents[2] / "fixtures" / "scicode_19.json"


def record() -> dict:
    """Load validation problem 19 exactly as published by SciCode."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_problem() -> None:
    problem = load_problem(record())

    assert problem.id == "19"
    assert problem.dependencies == "import numpy as np\nfrom scipy.linalg import sqrtm"
    assert [step.id for step in problem.steps] == ["19.1", "19.2"]
    assert problem.steps[0].function.startswith("def tensor()")
    assert len(problem.steps[0].tests) == 3


def test_official_prompt() -> None:
    problem = load_problem(record())
    code = "def tensor(*args):\n    return np.kron(*args)"
    prompt = render_prompt(problem, 1, (code,), with_background=True)

    assert code in prompt
    assert problem.steps[1].background in prompt
    assert problem.steps[0].tests[0] not in prompt


def test_unique_steps() -> None:
    data = record()
    data["sub_steps"][1]["step_number"] = "19.1"

    with pytest.raises(ValueError, match="duplicate step IDs"):
        load_problem(data)
