import json
from pathlib import Path

import h5py

from ddsr_bench.benchmarks.scicode.evaluation.verifier import verify


def _targets(path: Path) -> None:
    with h5py.File(path, "w") as file:
        for step, value in (("1.1", 3), ("1.2", 5)):
            group = file.create_group(f"{step}/test1")
            group.create_dataset("target", data=value)


def test_scicode_requires_every_step(tmp_path: Path) -> None:
    solution = tmp_path / "solution.py"
    solution.write_text(
        "def first():\n    return 3\n\ndef second():\n    return 4\n",
        encoding="utf-8",
    )
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {"id": "1.1", "tests": ["assert first() == target"]},
                {"id": "1.2", "tests": ["assert second() == target"]},
                {"id": "13.6", "tests": ["raise AssertionError"]},
            ]
        ),
        encoding="utf-8",
    )
    targets = tmp_path / "test_data.h5"
    _targets(targets)

    result = verify(solution, cases, targets, timeout=10)

    assert result["reward"] == 0
    assert result["status"] == "different"
    assert result["passed"] == 1
    assert result["total"] == 2
    assert result["steps"] == [
        {"id": "1.1", "status": "passed"},
        {"id": "1.2", "status": "failed"},
    ]


def test_missing_targets(tmp_path: Path) -> None:
    result = verify(
        tmp_path / "solution.py",
        tmp_path / "cases.json",
        tmp_path / "missing.h5",
    )

    assert result["reward"] == 0
    assert result["status"] == "error"
