import json
from dataclasses import asdict
from hashlib import sha256
from importlib import resources
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.frontierscience.data.loader import load_problem
from ddsr_bench.benchmarks.frontierscience.data.schemas import AnswerSpec
from ddsr_bench.benchmarks.frontierscience.evaluation.evaluator import (
    evaluate,
    parse_judgment,
)
from ddsr_bench.benchmarks.frontierscience.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.frontierscience.evaluation.static import run_trial
from ddsr_bench.benchmarks.frontierscience.generation.prompts import (
    judge_prompt,
    messages,
)
from ddsr_bench.generation.client import ChatResponse, Sampling
from tests.benchmarks.helpers import assert_static_pipeline


def _problem(track: str = "olympiad"):
    return load_problem(
        {
            "task_group_id": "p1",
            "subject": "physics",
            "problem": "Solve.\n\nThink step by step. FINAL ANSWER",
            "answer": "42" if track == "olympiad" else "Points: 10, Item: correct",
        },
        track,
    )


class FakeJudge:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    async def chat(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return ChatResponse(self.content, None, "judge", {}, None, 0.1, {}, "stop")

    async def preflight(self) -> None:
        pass


def test_public_projection_and_exact_request() -> None:
    problem = _problem()
    public = json.dumps(asdict(problem.spec))

    assert messages(problem.spec) == ({"role": "user", "content": problem.spec.prompt},)
    assert "42" not in public
    assert problem.spec.task_group_id == "p1"
    assert problem.answer == AnswerSpec("42")


def test_official_judge_prompts_and_parsers() -> None:
    olympiad = _problem()
    research = _problem("research")

    assert "GRADE: C" in judge_prompt(olympiad.spec, "42", olympiad.answer)
    assert "VERDICT: <total_points>" in judge_prompt(
        research.spec, "work", research.answer
    )
    assert parse_judgment("olympiad", "reason\nGRADE: C") == (1.0, True)
    assert parse_judgment("olympiad", "reason\nGRADE: I") == (0.0, False)
    assert parse_judgment("research", "reason\nVERDICT: 7.5") == (7.5, True)
    assert parse_judgment("research", "reason\nVERDICT: 6") == (6.0, False)
    assert parse_judgment("research", "VERDICT: nope") is None


def test_official_prompt_snapshots() -> None:
    root = resources.files("configs").joinpath("prompts", "frontierscience")
    assert sha256(root.joinpath("olympiad.txt").read_bytes()).hexdigest() == (
        "b74f497dd1574647a16e691ac1e74eefc6168997d489e66c5fdab7c7eb3deca0"
    )
    assert sha256(root.joinpath("research.txt").read_bytes()).hexdigest() == (
        "3131927d403e9372b26f647fa582ddc5f3f162731dade4e8da0c2ebfb6585b55"
    )


@pytest.mark.asyncio
async def test_track_specific_evaluation() -> None:
    problem = _problem("research")
    judge = FakeJudge("criteria\nVERDICT: 8.5")

    result = await evaluate("answer", problem.spec, problem.answer, judge)

    assert result["reward"] == 0.85
    assert result["points"] == 8.5
    assert result["status"] == "passed"
    assert judge.calls[0][1] == {"max_tokens": 4096}


@pytest.mark.asyncio
async def test_empty_response_skips_judge() -> None:
    problem = _problem()
    judge = FakeJudge("GRADE: C")

    result = await evaluate("  ", problem.spec, problem.answer, judge)

    assert result["error"] == "NoResponse"
    assert not judge.calls


def test_prepared_task_keeps_reference_private(tmp_path: Path) -> None:
    problem = _problem()
    task = compile_problem(problem, tmp_path)

    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    reference = (task / "tests" / "reference.json").read_text(encoding="utf-8")
    assert "42" not in instruction
    assert '"reference": "42"' in reference


@pytest.mark.asyncio
async def test_static_trial_keeps_reference_private(tmp_path: Path) -> None:
    problem = _problem()
    result = await run_trial(
        problem,
        0,
        tmp_path / "trial",
        FakeJudge("my answer"),
        FakeJudge("reason\nGRADE: C"),
        Sampling("teacher"),
        client_name="vllm",
        judge_name="openai",
        judge_model="gpt-5",
        judge_max_tokens=4096,
    )

    record = (tmp_path / "trial" / "agent" / "response.json").read_text(
        encoding="utf-8"
    )
    assert result["status"] == "passed"
    assert '"reference": "42"' not in record
    assert_static_pipeline(tmp_path / "trial", "frontierscience")
