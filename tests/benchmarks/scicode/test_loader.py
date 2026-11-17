import json
from pathlib import Path

import h5py
import pytest

from ddsr_bench.benchmarks.scicode.data import loader
from ddsr_bench.benchmarks.scicode.data.loader import (
    DATA_FILE,
    DATASET,
    REVISION,
    SPLIT,
    TARGETS_FILE,
    VERSION,
    load_problem,
    load_problems,
)
from ddsr_bench.benchmarks.scicode.generation.runner import render_prompt

FIXTURE = Path(__file__).parents[2] / "fixtures" / "scicode_19.json"


def record() -> dict:
    """Load one original SciCode record retained for schema compatibility."""
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


def test_fixed_class_step_allows_no_return_line() -> None:
    data = record()
    data["sub_steps"][0]["step_number"] = "13.6"
    data["sub_steps"][0]["return_line"] = None

    assert load_problem(data).steps[0].return_line == ""


def test_source_requires_matching_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = load_problem(record())
    monkeypatch.setattr(loader, "load_file", lambda path: [problem])
    target = tmp_path / TARGETS_FILE
    with h5py.File(target, "w") as file:
        for step in problem.steps:
            for index in range(1, len(step.tests) + 1):
                file.create_group(f"{step.id}/test{index}")

    loader.validate_source(tmp_path)
    with h5py.File(target, "a") as file:
        del file[f"{problem.steps[0].id}/test1"]
    with pytest.raises(ValueError, match="targets require"):
        loader.validate_source(tmp_path)


def test_unique_problems() -> None:
    with pytest.raises(ValueError, match="problem IDs must be unique"):
        load_problems([record(), record()])


def test_pinned_source() -> None:
    assert DATASET == "shhu2001/SciCode-Verified"
    assert REVISION == "752eb6b786b0e0d0a3b44cdb7c066c559b952658"
    assert VERSION == "v2.2"
    assert DATA_FILE == "data/problems_test.jsonl"
    assert TARGETS_FILE == "test_data_cleaned.h5"
    assert SPLIT == "test"
