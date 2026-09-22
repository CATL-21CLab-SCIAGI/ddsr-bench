import csv
import json
from typing import ClassVar

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus.batch import score_batch
from ddsr_bench.benchmarks.critpt.evaluation.consensus.candidates import load_candidates
from ddsr_bench.benchmarks.critpt.evaluation.local import score


class Grader:
    rows: ClassVar[dict] = {
        f"Challenge_{i}_main": {
            "id": f"Challenge_{i}_main",
            "mode": "skip" if i == 3 else "active",
            "note": "Excluded after review" if i == 3 else "",
        }
        for i in range(1, 4)
    }

    def grade(self, problem_id, code, *, input_error):
        if problem_id == "Challenge_3_main":
            status, matched = "skipped", None
        elif input_error:
            status, matched = "candidate_error", False
        elif code is None:
            status, matched = "missing_candidate", False
        elif "return 1" in code:
            status, matched = "matched", True
        else:
            status, matched = "unknown", None
        return {
            "problem_id": problem_id,
            "status": status,
            "matched": matched,
            "confidence": 1,
            "methods": [],
            "reference_coverage": "complete",
            **(
                {"skip_reason": self.rows[problem_id]["note"]}
                if status == "skipped"
                else {}
            ),
            **({"error": input_error} if input_error else {}),
        }


def native(directory, problem, attempt, answer=None, error=False):
    trial = directory / f"Challenge_{problem}_main__attempt-{attempt}"
    trial.mkdir(parents=True)
    result = {
        "task_name": f"critpt/Challenge_{problem}_main",
        "attempt": attempt,
        "static_result": {
            "status": "error" if error else "validated",
            "message": "format failed",
        },
    }
    (trial / "result.json").write_text(json.dumps(result))
    if answer:
        (trial / "artifacts").mkdir()
        (trial / "artifacts/answer.py").write_text(answer)


def test_batch_retains_attempts_errors_unknowns_and_fixed_denominator(tmp_path):
    source, output = tmp_path / "job", tmp_path / "grade"
    native(source, 1, 0, "def answer(): return 1")
    native(source, 1, 2, "def answer(): return 0")
    native(source, 2, 0, error=True)
    summary = score_batch(source, output, Grader(), jobs=2, metadata={})
    assert summary["active_trials"] == 4
    assert summary["matched_trials"] == 1
    assert summary["mean_match_rate"] == 0.25
    assert summary["skipped_trials"] == 2
    assert [a["attempt"] for a in summary["attempts"]] == [0, 2]
    assert len(list((output / "trials").glob("*.json"))) == 6
    rows = list(csv.DictReader((output / "attempt-results.csv").open()))
    assert len(rows) == 6
    assert next(r for r in rows if r["status"] == "unknown")["score"] == "0"
    skipped = next(r for r in rows if r["status"] == "skipped")
    assert skipped["score"] == ""
    assert skipped["skip_reason"] == "Excluded after review"
    assert summary["problems"][2]["skip_reason"] == "Excluded after review"
    assert (
        next(r for r in rows if r["status"] == "candidate_error")["error_stage"]
        == "generation"
    )
    assert json.loads((output / "progress.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        score_batch(source, output, Grader(), jobs=1, metadata={})


def test_batch_keeps_completed_trials_if_later_grading_fails(tmp_path):
    class Broken(Grader):
        def grade(self, problem_id, code, *, input_error):
            if problem_id == "Challenge_2_main":
                raise RuntimeError("interrupted scoring")
            return super().grade(problem_id, code, input_error=input_error)

    source, output = tmp_path / "job", tmp_path / "grade"
    native(source, 1, 0, "def answer(): return 1")
    with pytest.raises(RuntimeError, match="interrupted"):
        score_batch(source, output, Broken(), jobs=1, metadata={})
    assert (output / "trials/Challenge_1_main__attempt-0.json").is_file()
    assert not (output / "summary.json").exists()


def test_batch_reports_oversized_answer_without_hashing_unbounded_input(tmp_path):
    source, output = tmp_path / "job", tmp_path / "grade"
    native(source, 1, 0, "x" * 8_000_001)
    score_batch(source, output, Grader(), jobs=1, metadata={})
    trial = json.loads((output / "trials/Challenge_1_main__attempt-0.json").read_text())
    assert trial["status"] == "candidate_error"
    assert trial["error"]["stage"] == "input"
    assert trial["candidate_sha256"] is None


def test_single_and_batch_reports(tmp_path):
    source, output = tmp_path / "job", tmp_path / "grade"
    native(source, 1, 0, "def answer(): return 1")
    native(source, 2, 0, error=True)
    native(source, 1, 2, "def answer(): return 0")
    agent = source / "Challenge_1_main__attempt-0" / "agent"
    agent.mkdir()
    (agent / "response.json").write_text(
        json.dumps(
            {"problem_id": "Challenge_1_main", "model": "teacher", "responses": []}
        )
    )
    score_batch(source, output, Grader(), jobs=2, metadata={})
    for attempt in (0, 2):
        single = score(source, load_candidates(source, attempt), Grader(), jobs=2)
        batch = json.loads((output / f"attempt-{attempt}.json").read_text())
        assert batch["candidate_input"] == single["candidate_input"]
        assert batch["summary"] == single["summary"]
        for row, expected in zip(batch["results"], single["results"], strict=True):
            assert row.pop("model") == source.name
            assert row.pop("attempt") == attempt
            if "candidate_source" in expected:
                assert "candidate_sha256" in row
                row.pop("candidate_sha256")
            assert row == expected
