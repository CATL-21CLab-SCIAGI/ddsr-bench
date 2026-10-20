"""Collection-to-grader contracts; optional Docker checks, no live judge calls."""

import asyncio
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from harbor.models.job.config import AgentConfig
from omegaconf import OmegaConf

from ddsr_bench.benchmarks.collect import collect_trials, load_batch
from ddsr_bench.benchmarks.critpt.data.loader import load_ai_task
from ddsr_bench.benchmarks.critpt.evaluation.ai import answer, grader
from ddsr_bench.benchmarks.critpt.evaluation.ai.llm.semantic import (
    LLMSemanticJudge,
    SemanticJudgeConfig,
)
from ddsr_bench.benchmarks.critpt.evaluation.prepare import decode_instruction
from ddsr_bench.benchmarks.critpt.evaluation.static import run_trial
from ddsr_bench.benchmarks.utils import read_json, write_json
from ddsr_bench.generation.client import ChatResponse

from . import test_task_inputs

prepared = test_task_inputs.prepared


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Network and answer-code execution are forbidden")

    monkeypatch.setattr(answer, "_execute_answer_code", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["pass", "fail", "error"])
async def test_collected(prepared, tmp_path, monkeypatch, outcome):
    def native_parser(*args, **kwargs):
        pytest.fail("Prepared tasks must not use the native compatibility parser")

    monkeypatch.setattr(grader, "parse_grading_problem", native_parser)
    task, spec = prepared
    if spec.grader == "rule":
        reference = task / "tests/reference.json"
        data = read_json(reference)
        data["grading_plan"]["calls"][0]["expected"] = "2"
        write_json(reference, data)
    problem = load_ai_task(task)
    response = (
        "Derivation.\n```python\ndef answer():\n    return 2\n```"
        if spec.grader == "rule"
        else "Derivation and final prose answer."
    )

    class Teacher:
        async def chat(self, messages):
            return ChatResponse(response, "", "mock", {}, None, 0.1, {}, "stop")

    job = tmp_path / "job"
    for attempt in (0, 1):
        await run_trial(
            spec,
            attempt,
            job / f"{spec.id}__attempt-{attempt}",
            Teacher(),
            AgentConfig(model_name="mock"),
        )
    originals = {p: p.read_bytes() for p in job.rglob("*") if p.is_file()}
    collect_trials(job)
    summary = (job / "summary.json").read_bytes()

    # Stub only the execution boundary; exercise real deterministic comparison.
    extraction = Mock(return_value={"answer": 2 if outcome == "pass" else 3})
    if outcome == "error":
        extraction.side_effect = RuntimeError("Synthetic execution failure")
    monkeypatch.setattr(grader, "extract_answer_locally", extraction)

    def complete(messages, config):
        if outcome == "error":
            raise RuntimeError("Synthetic provider failure")
        return json.dumps(
            {
                "verdict": "correct" if outcome == "pass" else "incorrect",
                "confidence": 1.0,
                "solver_answer_summary": "Synthetic answer",
                "assessment": "Synthetic assessment",
                "substantive_errors": [],
            }
        )

    judge = LLMSemanticJudge(
        config=SemanticJudgeConfig(model="mock"), completion=complete
    )
    for attempt in (0, 1):
        (trial,) = load_batch(job, attempt)
        assert trial["problem_id"] == spec.id != problem.answer.problem_id
        assert trial["attempt"] == attempt and trial["reward"] is None
        record = read_json(job / trial["trial_name"] / "agent/response.json")
        saved_response = record["messages"][-1]["content"]
        assert saved_response == response
        artifact = (job / trial["artifact"]).read_text()
        assert (artifact != response) is (spec.grader == "rule")
        report = grader.grade_response(
            problem,
            saved_response,
            attempt_id=trial["trial_name"],
            judge_enabled=False,
            semantic_judge=judge,
        )
        assert report.problem_id == problem.answer.problem_id
        assert report.response_sha256 == hashlib.sha256(response.encode()).hexdigest()
        assert report.grading_status == (
            "grader_error" if outcome == "error" else outcome
        )
        if outcome == "error":
            assert report.score is None
        if spec.grader == "llm" and outcome != "error":
            assert report.semantic_judgment.attempt_id == trial["trial_name"]
    assert extraction.call_count == (2 if spec.grader == "rule" else 0)
    assert all(p.read_bytes() == content for p, content in originals.items())
    assert (job / "summary.json").read_bytes() == summary


def test_saved_collection(tmp_path, monkeypatch):
    """Opt-in real saved data: validate joining and response fidelity, not scores."""
    source = os.getenv("DDSR_TEST_AI_JOB")
    tasks = os.getenv("DDSR_TEST_AI_TASKS")
    if not source or not tasks:
        pytest.skip("Set DDSR_TEST_AI_JOB and DDSR_TEST_AI_TASKS")
    source = Path(source)
    job = tmp_path / "job"
    job.mkdir()
    selected = {}
    for path in sorted(source.glob("*/result.json")):
        result = read_json(path)
        if result.get("source") != "critpt-ai":
            continue
        kind = result.get("grader")
        if kind in selected or not (path.parent / "agent/response.json").is_file():
            continue
        selected[kind] = path.parent
        shutil.copytree(path.parent, job / path.parent.name)
    assert set(selected) == {"rule", "llm"}, "Need saved responses for both graders"
    summary = collect_trials(job)
    for batch in summary["batches"]:
        for trial in load_batch(job, batch["attempt"]):
            matches = list(Path(tasks).rglob(f"{trial['problem_id']}/instruction.md"))
            assert len(matches) == 1
            spec = decode_instruction(matches[0].read_text())
            problem = load_ai_task(matches[0].parent)
            record = read_json(job / trial["trial_name"] / "agent/response.json")
            original = read_json(selected[spec.grader] / "agent/response.json")
            assert record == original
            assert record["messages"][-1]["role"] == "assistant"
            assert record["messages"][-1]["content"]
            assert spec.id == trial["problem_id"]
            assert problem.spec.statement == spec.statement
            assert trial["benchmark_config"]["source"] == "critpt-ai"
            assert trial["reward"] is None

    image = os.getenv("DDSR_TEST_AI_IMAGE")
    if not image:
        return
    from ddsr_bench.benchmarks.critpt.submission import ai, submit

    # Replay untouched saved responses: real isolated rule execution, no paid judge.
    originals = {p: p.read_bytes() for p in job.rglob("*") if p.is_file()}
    assessed = []

    def complete(messages, config):
        assessed.append(json.loads(messages[-1]["content"])["solver_attempt"])
        return json.dumps(
            {
                "verdict": "correct",
                "confidence": 1,
                "solver_answer_summary": "Mocked assessment",
                "assessment": "Transport test only, not a correctness claim",
                "substantive_errors": [],
            }
        )

    monkeypatch.setattr(
        ai,
        "LLMSemanticJudge",
        lambda config: LLMSemanticJudge(config=config, completion=complete),
    )
    attempts = [batch["attempt"] for batch in summary["batches"]]
    submit(
        job,
        {
            "backend": "internal",
            "attempts": attempts,
            "datasets": [{"path": str(Path(tasks) / kind)} for kind in ("rule", "llm")],
            "ai": {
                "llm": {"model": "mock"},
                "resources": {
                    "image": image,
                    "cpus": 1,
                    "memory_mb": 1024,
                    "timeout_sec": 120,
                },
            },
        },
    )
    label = "-".join(map(str, sorted(attempts)))
    report = read_json(job / f"submission-internal-{label}.json")
    rows = {row["trial_name"]: row for row in report["results"]}
    assert set(rows) == {path.name for path in selected.values()}
    for kind, path in selected.items():
        row = rows[path.name]
        record = read_json(path / "agent/response.json")
        response = record["responses"][-1]["content"]
        assert row["problem_id"] == record["problem_id"]
        assert row["attempt"] == read_json(path / "result.json")["attempt"]
        assert (
            row["report"]["response_sha256"]
            == hashlib.sha256(response.encode()).hexdigest()
        )
        if kind == "rule":
            assert row["report"]["grading_status"] in {"pass", "fail"}
        else:
            assert assessed == [{"attempt_id": path.name, "response": response}]
    assert all(p.read_bytes() == content for p, content in originals.items())
    assert not list(job.glob("*.pending.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["job-config.yaml", "config.json", "override"])
@pytest.mark.parametrize("worker_result", ["pass", "fail", "timeout", "error"])
async def test_submission(prepared, tmp_path, monkeypatch, location, worker_result):
    from ddsr_bench.benchmarks.critpt.evaluation.ai.runtime import GraderTimeout
    from ddsr_bench.benchmarks.critpt.submission import ai, submit

    task, spec = prepared
    if spec.grader == "rule":
        reference = task / "tests/reference.json"
        data = read_json(reference)
        data["grading_plan"]["calls"][0]["expected"] = "2"
        write_json(reference, data)

    content = (
        "Intermediate calculation.\n```python\nscratch = 1\n```\n"
        "Final answer.\n```python\ndef answer():\n    return 2\n```"
    )

    class Teacher:
        async def chat(self, messages):
            return ChatResponse(content, "", "mock", {}, None, 0.1, {}, "stop")

    job = tmp_path / "job"
    for attempt in (0, 1):
        await run_trial(
            spec,
            attempt,
            job / f"trial-{attempt}",
            Teacher(),
            AgentConfig(model_name="mock"),
        )
        if spec.grader == "rule":
            # Official extraction sees the first block; AI grading must still
            # assess the saved response using its own last-answer-block rule.
            assert (
                read_json(job / f"trial-{attempt}/validation/result.json")["status"]
                == "error"
            )
            assert not (job / f"trial-{attempt}/artifacts/answer.py").exists()
    calls = []

    def complete(messages, config):
        assert config.model == "assessment-model"
        assert (
            json.loads(messages[-1]["content"])["solver_attempt"]["response"] == content
        )
        calls.append(messages)
        return json.dumps(
            {
                "verdict": "correct",
                "confidence": 1,
                "solver_answer_summary": "answer",
                "assessment": "correct",
                "substantive_errors": [],
            }
        )

    factory = Mock(
        side_effect=lambda config: LLMSemanticJudge(config=config, completion=complete)
    )
    monkeypatch.setattr(ai, "LLMSemanticJudge", factory)
    worker = Mock(return_value={"answer": 2 if worker_result == "pass" else 3})
    if worker_result in {"timeout", "error"}:
        worker.side_effect = (
            GraderTimeout()
            if worker_result == "timeout"
            else RuntimeError("worker failed")
        )
    monkeypatch.setattr(ai, "execute", worker)
    config = {
        "backend": "internal",
        "attempts": [1, 0],
        "ai": {
            "llm": {"model": "assessment-model"},
            "resources": {
                "image": "mock",
                "cpus": 1,
                "memory_mb": 512,
                "timeout_sec": 10,
            },
        },
    }
    datasets = [{"path": str(task.parent)}]
    if location == "override":
        config["datasets"] = datasets
        write_json(job / "job-config.yaml", {"datasets": [{"path": "unavailable"}]})
    else:
        write_json(job / location, {"datasets": datasets})
    config = OmegaConf.create(config)
    # Preflight every attempt before spending judge calls.
    record_path = job / "trial-1/agent/response.json"
    saved = record_path.read_bytes()
    record_path.unlink()
    with pytest.raises(ValueError, match="cannot read"):
        submit(job, config)
    assert not calls and not list(job.glob("*.pending.json"))
    record_path.write_bytes(saved)
    summary = (job / "summary.json").read_bytes()
    submit(job, config)
    report = read_json(job / "submission-internal-0-1.json")
    assert len(calls) == (2 if spec.grader == "llm" else 0)
    assert worker.call_count == (2 if spec.grader == "rule" else 0)
    if spec.grader == "llm":
        factory.assert_called_once()
    else:
        factory.assert_not_called()
        assert all(set(call.kwargs) == {"resources"} for call in worker.call_args_list)
        assert all(
            "def answer()" in call.args[0] and "scratch" not in call.args[0]
            for call in worker.call_args_list
        )
    assert [r["attempt"] for r in report["results"]] == [0, 1]
    assert all(r["problem_id"] == spec.id for r in report["results"])
    assert all(r["report"]["problem_id"] == "native-id" for r in report["results"])
    expected = (
        "pass"
        if spec.grader == "llm"
        else (
            "grader_error" if worker_result in {"timeout", "error"} else worker_result
        )
    )
    assert all(r["report"]["grading_status"] == expected for r in report["results"])
    if expected == "grader_error":
        assert all(r["report"]["score"] is None for r in report["results"])
    assert (job / "summary.json").read_bytes() == summary
    assert record_path.read_bytes() == saved
    assert not list(job.glob("*.pending.json"))
    with pytest.raises(ValueError, match="already exists"):
        submit(job, config)
    assert len(calls) == (2 if spec.grader == "llm" else 0)


def test_dataset_paths(tmp_path, monkeypatch):
    from ddsr_bench.benchmarks.critpt.submission.ai import _task_paths

    job = tmp_path / "job"
    job.mkdir()
    with pytest.raises(ValueError, match="saved job config missing"):
        _task_paths(job, None)
    # Same relative path semantics as generation, not relative to the job directory.
    monkeypatch.chdir(tmp_path)
    task = tmp_path / "tasks" / "one"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("{}")
    write_json(job / "job-config.yaml", {"datasets": [{"path": "tasks"}]})
    assert _task_paths(job, None) == [Path("tasks/one/instruction.md")]
    for datasets in ([], [{"path": "missing"}]):
        with pytest.raises(ValueError, match="datasets"):
            _task_paths(job, datasets)


def test_mixed_sources(tmp_path):
    from ddsr_bench.benchmarks.critpt.submission import submit

    batches = [
        {
            "attempt": i,
            "trials": [
                {
                    "attempt": i,
                    "problem_id": "one",
                    "benchmark_config": {"source": source},
                }
            ],
        }
        for i, source in enumerate(("critpt-ai", "critpt-official"))
    ]
    write_json(tmp_path / "summary.json", {"batches": batches})
    with pytest.raises(ValueError, match="one collected data source"):
        submit(tmp_path, {"attempts": [0, 1], "backend": "internal"})
    assert not list(tmp_path.glob("*.pending.json"))


@pytest.mark.parametrize("live", [False, True])
def test_mixed_graders(prepared, tmp_path, monkeypatch, live):
    from ddsr_bench.benchmarks.critpt.data.schemas import AIProblem
    from ddsr_bench.benchmarks.critpt.evaluation.prepare import compile_problem
    from ddsr_bench.benchmarks.critpt.submission import ai, submit
    from ddsr_bench.benchmarks.utils import Resources

    if live and not os.getenv("DDSR_TEST_AI_IMAGE"):
        pytest.skip("Set DDSR_TEST_AI_IMAGE for live rule submission")
    task, _ = prepared
    original = load_ai_task(task)
    limits = {
        "image": os.getenv("DDSR_TEST_AI_IMAGE", "mock"),
        "cpus": 1,
        "memory_mb": 1024,
        "timeout_sec": 30,
    }
    completion = Mock(
        return_value=json.dumps(
            {
                "verdict": "correct",
                "confidence": 1,
                "solver_answer_summary": "2",
                "assessment": "Mocked assessment",
                "substantive_errors": [],
            }
        )
    )
    monkeypatch.setattr(
        ai,
        "LLMSemanticJudge",
        lambda config: LLMSemanticJudge(config=config, completion=completion),
    )
    if not live:
        monkeypatch.setattr(ai, "execute", Mock(return_value={"answer": 2}))
    job = tmp_path / "mixed"

    class Teacher:
        async def chat(self, messages):
            return ChatResponse(
                "```python\ndef answer():\n    return 2\n```",
                "",
                "mock",
                {},
                None,
                0.1,
                {},
                "stop",
            )

    for kind in ("rule", "llm"):
        spec = replace(
            original.spec,
            id=f"ai-{kind}",
            grader=kind,
            code_template="def answer():\n    return ..." if kind == "rule" else "",
        )
        reference = replace(
            original.answer,
            snippet="2",
            answer_type="exact",
            answer_items=[
                {"item_id": "value", "answer_type": "exact", "target": "value"}
            ],
            answer_contract={"answer_type": "exact"},
            grading_plan={
                "calls": [{"primitive": "exact", "target": "answer", "expected": "2"}]
            },
        )
        compile_problem(AIProblem(spec, reference), task.parent, Resources(**limits))
        asyncio.run(
            run_trial(spec, 0, job / kind, Teacher(), AgentConfig(model_name="mock"))
        )
    write_json(job / "job-config.yaml", {"datasets": [{"path": str(task.parent)}]})
    submit(
        job,
        {
            "backend": "internal",
            "attempts": [0],
            "ai": {"llm": {"model": "mock"}, "resources": limits},
        },
    )
    result = read_json(job / "submission-internal-0.json")
    assert {row["problem_id"] for row in result["results"]} == {"ai-rule", "ai-llm"}
    assert all(row["report"]["grading_status"] == "pass" for row in result["results"])
    completion.assert_called_once()
