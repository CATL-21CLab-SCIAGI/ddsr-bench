import json
from hashlib import sha256
from pathlib import Path

import pytest
from harbor.models.job.config import AgentConfig

from ddsr_bench.benchmarks.critpt.data.schemas import ProblemSpec
from ddsr_bench.benchmarks.critpt.evaluation.consensus.candidates import (
    MAX_BYTES,
    candidate_digest,
    load_candidates,
)
from ddsr_bench.benchmarks.critpt.evaluation.consensus.cli import main
from ddsr_bench.benchmarks.critpt.evaluation.consensus.grader import Grader
from ddsr_bench.benchmarks.critpt.evaluation.static import run_trial
from ddsr_bench.generation.client import ChatResponse


@pytest.mark.parametrize("payload", [b"", b"\xff", b"x" * MAX_BYTES])
def test_candidate_digest(tmp_path, payload):
    path = tmp_path / "answer.py"
    path.write_bytes(payload)
    assert candidate_digest(path) == sha256(payload).hexdigest()


def test_unhashable_candidate(tmp_path):
    path = tmp_path / "answer.py"
    assert candidate_digest(path) is None
    assert candidate_digest(tmp_path) is None
    path.write_bytes(b"x" * (MAX_BYTES + 1))
    assert candidate_digest(path) is None


@pytest.mark.parametrize(
    "payload",
    [
        b"{",
        b"[]",
        b"{}",
        b'{"generated_code":null}',
        b'{"generated_code":7}',
        b'{"generated_code":"  "}',
        b'{"generated_code":"```python\\n```"}',
        b'{"generated_code":NaN}',
        b'{"problem_id":"Challenge_2_main","generated_code":"def answer(): return 1"}',
        b"\xff",
    ],
)
def test_bad_candidate_does_not_discard_other_answers(tmp_path, payload):
    (tmp_path / "Challenge_1_main.json").write_bytes(payload)
    (tmp_path / "Challenge_2_main.py").write_text("def answer(): return 2")
    answers = load_candidates(tmp_path).answers
    assert answers["Challenge_1_main"].error["stage"] == "input"
    assert answers["Challenge_2_main"].code == "def answer(): return 2\n"


def test_wrong_paths_and_empty_discovery_fail_instead_of_reporting_zero(
    tmp_path, sample_bundle_path
):
    for path in (tmp_path / "missing", tmp_path):
        with pytest.raises(ValueError):
            load_candidates(path)
    (tmp_path / "response.json").write_text("{}")
    with pytest.raises(ValueError, match="no recognized"):
        load_candidates(tmp_path)
    assert (
        main(
            [
                "score",
                "--bundle",
                str(sample_bundle_path),
                "--candidates",
                str(tmp_path),
                "--output",
                str(tmp_path / "score.json"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "score.json").exists()


def native(root: Path, number: int, attempt: int, code: str | None):
    trial = root / f"Challenge_{number}_main__attempt-{attempt}"
    (trial / "artifacts").mkdir(parents=True)
    if code is not None:
        (trial / "artifacts" / "answer.py").write_text(code)
    return trial


def test_attempt_selection_never_borrows_another_attempt(tmp_path):
    native(tmp_path, 1, 0, "def answer(): return 1")
    native(tmp_path, 2, 0, "def answer(): return 2")
    native(tmp_path, 1, 1, "def answer(): return 3")
    with pytest.raises(ValueError, match="--attempt"):
        load_candidates(tmp_path)
    batch = load_candidates(tmp_path, 1)
    assert batch.attempt == 1 and batch.available_attempts == [0, 1]
    assert list(batch.answers) == ["Challenge_1_main"]
    assert batch.answers["Challenge_1_main"].code.endswith("return 3\n")
    with pytest.raises(ValueError, match="not found"):
        load_candidates(tmp_path, 9)


def test_mixed_jobs_and_layouts_are_rejected(tmp_path):
    native(tmp_path / "job-a", 1, 0, "def answer(): return 1")
    native(tmp_path / "job-b", 2, 0, "def answer(): return 2")
    with pytest.raises(ValueError, match="multiple rollout job"):
        load_candidates(tmp_path)
    (tmp_path / "Challenge_3_main.py").write_text("def answer(): return 3")
    with pytest.raises(ValueError, match="cannot combine"):
        load_candidates(tmp_path)


def test_native_missing_failed_and_inconsistent_trials(tmp_path):
    native(tmp_path, 1, 0, None)
    failed = native(tmp_path, 2, 0, None)
    (failed / "result.json").write_text(
        json.dumps(
            {
                "task_name": "critpt/Challenge_2_main",
                "attempt": 0,
                "static_result": {
                    "status": "error",
                    "error": "ClientError",
                    "message": "empty response",
                },
            }
        )
    )
    mismatch = native(tmp_path, 3, 0, "def answer(): return 3")
    (mismatch / "result.json").write_text(
        json.dumps({"task_name": "critpt/Challenge_4_main", "attempt": 0})
    )
    answers = load_candidates(tmp_path).answers
    assert answers["Challenge_1_main"].code is None
    assert answers["Challenge_1_main"].error is None
    assert answers["Challenge_2_main"].error["stage"] == "generation"
    assert answers["Challenge_2_main"].error["upstream"]["error"] == "ClientError"
    assert answers["Challenge_3_main"].error["stage"] == "input"


def test_corrupt_optional_response_metadata_does_not_reject_valid_code(tmp_path):
    trial = native(tmp_path, 1, 0, "def answer(): return 1")
    (trial / "agent").mkdir()
    (trial / "agent" / "response.json").write_text("{")
    candidate = load_candidates(trial).answers["Challenge_1_main"]
    assert candidate.code == "def answer(): return 1\n"
    assert candidate.error is None
    assert candidate.metadata_errors


def test_input_errors_do_not_execute_references_and_skip_stays_skip(sample_bundle):
    class NoRun:
        def run(self, _):
            raise AssertionError("invalid input or skip executed")

    bundle = sample_bundle
    grader = Grader(bundle, NoRun())
    error = {"stage": "input", "error": "invalid JSON"}
    assert (
        grader.grade("Challenge_1_main", None, input_error=error)["status"]
        == "candidate_error"
    )
    assert (
        grader.grade("Challenge_6_main", None, input_error=error)["status"] == "skipped"
    )


@pytest.mark.asyncio
async def test_native_rollout_to_score_keeps_two_step_and_failure_diagnostics(
    tmp_path, sample_bundle, sample_bundle_path
):
    bundle = sample_bundle
    reference = bundle["problems"][0]["references"][0]["code"]

    class Client:
        def __init__(self, texts):
            self.texts = iter(texts)
            self.calls = []

        async def chat(self, messages, **kwargs):
            self.calls.append(messages)
            text = next(self.texts)
            return ChatResponse(
                text,
                "reasoning",
                "mock",
                {"completion_tokens": 10},
                None,
                0.1,
                {},
                finish_reason="stop" if text else "length",
            )

    agent = AgentConfig(
        name="ddsr_bench.benchmarks.critpt.evaluation.harbor:CritPtAgent",
        model_name="mock",
        kwargs={"style": "two-step"},
    )
    job = tmp_path / "job"
    for number, client in [
        (1, Client(["", reference])),
        (2, Client(["derivation", ""])),
    ]:
        row = bundle["problems"][number - 1]
        problem = ProblemSpec(
            row["id"],
            "main",
            None,
            "Mock problem.",
            row["template"],
            "critpt",
            Path("unused.json"),
        )
        status = await run_trial(
            problem, 0, job / f"{row['id']}__attempt-0", client, agent
        )
        assert len(client.calls) == 2
        assert status == ("validated" if number == 1 else "error")
    output = tmp_path / "score.json"
    assert (
        main(
            [
                "score",
                "--bundle",
                str(sample_bundle_path),
                "--candidates",
                str(job),
                "--output",
                str(output),
                "--trusted-local",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["candidate_input"]["attempt"] == 0
    assert report["summary"]["statuses"] == {
        "matched": 1,
        "candidate_error": 1,
        "missing_candidate": 59,
        "skipped": 9,
    }
    assert report["summary"]["active"] == 61
    good, bad = report["results"][:2]
    assert good["generation"]["responses"][0]["finish_reason"] == "length"
    assert good["generation"]["responses"][0]["content_empty"] is True
    assert good["generation"]["responses"][1]["finish_reason"] == "stop"
    assert bad["error"]["stage"] == "generation"
    assert bad["generation"]["responses"][1]["content_empty"] is True


def test_bad_flat_input_does_not_abort_score_report(
    tmp_path, sample_bundle, sample_bundle_path
):
    answers = tmp_path / "answers"
    answers.mkdir()
    (answers / "Challenge_1_main.json").write_text("{")
    reference = sample_bundle["problems"][16]["references"][0]["code"]
    (answers / "Challenge_17_main.py").write_text(reference)
    output = tmp_path / "score.json"
    assert (
        main(
            [
                "score",
                "--bundle",
                str(sample_bundle_path),
                "--candidates",
                str(answers),
                "--output",
                str(output),
                "--trusted-local",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["results"][0]["status"] == "candidate_error"
    assert report["results"][16]["status"] == "matched"
    assert report["summary"]["active"] == 61
