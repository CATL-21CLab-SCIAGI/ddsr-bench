from unittest.mock import Mock

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus.grader import grade


def test_saved_code(tmp_path):
    path = tmp_path / "answer.py"
    code = "def answer(): return 1\n"
    path.write_text(code)
    grader = Mock()
    grader.grade.return_value = {"status": "matched"}

    result = grade(grader, "Challenge_1_main", path, "validated")

    grader.grade.assert_called_once_with("Challenge_1_main", code, input_error=None)
    assert result["answer_source"] == str(path)
    assert result["generation"] is None


@pytest.mark.parametrize("status,stage", [("missing", None), ("error", "generation")])
def test_missing(status, stage):
    grader = Mock()
    grader.grade.return_value = {}
    grade(grader, "Challenge_1_main", None, status)
    error = grader.grade.call_args.kwargs["input_error"]
    assert (error["stage"] if error else None) == stage


@pytest.mark.parametrize("content", [None, b"", b"\xff"])
def test_unreadable(tmp_path, content):
    path = tmp_path / "answer.py"
    if content is not None:
        path.write_bytes(content)
    grader = Mock()
    grader.grade.return_value = {}
    grade(grader, "Challenge_1_main", path, "validated")
    assert grader.grade.call_args.kwargs["input_error"]["stage"] == "input"


def test_failure_details():
    grader = Mock()
    grader.grade.return_value = {}
    failure = {"status": "error", "message": "answer must define exactly one answer()"}
    grade(grader, "Challenge_1_main", None, "error", upstream=failure)
    assert grader.grade.call_args.kwargs["input_error"] == {
        "status": "error",
        "stage": "generation",
        "error": "rollout did not produce an answer artifact",
        "upstream": failure,
    }
