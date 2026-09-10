import json
from dataclasses import asdict
from pathlib import Path

import pytest

from critpt_eval.loaders import load_challenge, load_challenges, load_problems

EXAMPLE = Path(__file__).parent.parent / "fixtures" / "quantum_error_correction.json"
SINGLE = (
    Path(__file__).parent.parent / "fixtures" / "quantum_error_correction_main.json"
)


def write_fixture(path: Path, number: int, *, problem_type: str = "main") -> None:
    """Write the smallest valid CritPt challenge used by these tests."""
    payload = {
        "dataset_name": f"Challenge_{number}",
        "source_notebook": "ignored.ipynb",
        "problems": [
            {
                "problem_id": f"Challenge_{number}_main",
                "problem_type": problem_type,
                "problem_index": 0 if problem_type == "sub" else None,
                "problem_description": f"statement {number}",
                "code_template": "def answer():\n    return ...",
                "answer_code": "SECRET",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loads_public_problem(tmp_path: Path) -> None:
    path = tmp_path / "Challenge_1.json"
    write_fixture(path, 1)

    problem = load_challenge(path).main

    assert problem.spec.id == "Challenge_1_main"
    assert problem.spec.type == "main"
    assert problem.spec.index is None
    assert problem.spec.statement == "statement 1"
    assert problem.spec.source == "critpt"
    assert problem.spec.source_path == path
    assert problem.answer is None
    assert "SECRET" not in repr(problem)


def test_batch_is_sorted_numerically(tmp_path: Path) -> None:
    nested = tmp_path / "json"
    nested.mkdir()
    write_fixture(nested / "Challenge_10.json", 10)
    write_fixture(nested / "Challenge_2.json", 2)

    challenges = load_challenges(tmp_path)

    assert [challenge.main.spec.id for challenge in challenges] == [
        "Challenge_2_main",
        "Challenge_10_main",
    ]


def test_requires_main(tmp_path: Path) -> None:
    path = tmp_path / "Challenge_1.json"
    write_fixture(path, 1, problem_type="sub")

    with pytest.raises(ValueError, match="exactly one main"):
        load_challenge(path)


def test_subproblems() -> None:
    challenge = load_challenge(EXAMPLE)
    problems = list(challenge.problems)

    assert challenge.id == "quantum_error_correction"
    assert [problem.spec.id for problem in problems] == [
        "quantum_error_correction_main",
        "quantum_error_correction_sub_0",
        "quantum_error_correction_sub_1",
        "quantum_error_correction_sub_2",
    ]
    assert [problem.spec.index for problem in problems] == [None, 0, 1, 2]
    assert challenge.main == problems[0]

    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert load_problems(data, EXAMPLE) == challenge.problems


def test_unique_subproblem_indices(tmp_path: Path) -> None:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["problems"][-1]["problem_index"] = 1
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="indices must be unique"):
        load_challenge(path)


def test_answer_projection() -> None:
    problem = load_challenge(SINGLE).main

    assert problem.spec.id == "quantum_error_correction_main_main"
    assert problem.answer is not None
    assert "def real_answer" in problem.answer.code
    assert problem.answer.snippet.startswith("F_logical =")
    assert problem.answer.testcases is None


def test_public_projection() -> None:
    problem = load_challenge(SINGLE).main
    public = json.dumps(asdict(problem.spec), default=str)

    assert "real_answer" not in public
    assert "answer_only_code" not in public


def test_single_problem_challenge() -> None:
    problems = load_challenge(SINGLE).problems

    assert [problem.spec.id for problem in problems] == [
        "quantum_error_correction_main_main"
    ]
