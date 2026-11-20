import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.jeebench.data.loader import load_problem
from ddsr_bench.benchmarks.jeebench.evaluation.evaluator import (
    evaluate,
    parse_answer,
    score,
)
from ddsr_bench.benchmarks.jeebench.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.jeebench.evaluation.static import run_trial
from ddsr_bench.benchmarks.jeebench.generation.prompts import messages, render
from ddsr_bench.generation.client import ChatResponse, Sampling


def _problem(response_type: str = "MCQ(multiple)", gold: str = "AC"):
    return load_problem(
        {
            "description": "JEE Adv 2023 Paper 1",
            "index": 7,
            "subject": "phy",
            "type": response_type,
            "question": "Which statements are true?\n\n(A) One\n\n(B) Two",
            "gold": gold,
        }
    )


def test_projection_and_official_prompt() -> None:
    problem = _problem()
    public = json.dumps(asdict(problem.spec))
    expected = (
        "In this problem, multiple options can be correct. Give a detailed solution "
        "and end the solution with the final answer.\n\nProblem: Which statements "
        "are true?\n(A) One\n(B) Two\nSolution: Let's think step by step."
    )

    assert '"gold"' not in public
    assert render(problem.spec) == expected
    assert messages(problem.spec)[1]["content"] == expected


def test_official_scoring_and_parser() -> None:
    problem = _problem()

    assert parse_answer("Final answer: (A), (C)", problem.spec.response_type) == "AC"
    assert (
        evaluate("Final answer: (A), (C)", problem.spec, problem.answer)["reward"] == 1
    )
    assert score("AC", "A", "MCQ(multiple)") == 0.25
    assert score("AC", "AB", "MCQ(multiple)") == 0
    assert score("1.00", "1.009", "Numeric") == 1
    assert score("1.00", "1.011", "Numeric") == 0
    assert parse_answer(r"Final answer: \frac{1}{4}", "Numeric") == "0.25"
    assert parse_answer(r"Final answer: 2.5 \times 10^{-2}", "Numeric") == "0.025"


def test_prepared_task_keeps_reference_private(tmp_path: Path) -> None:
    problem = _problem()
    task = compile_problem(problem, tmp_path)

    assert '"value"' not in (task / "instruction.md").read_text(encoding="utf-8")
    assert '"value": "AC"' in (task / "tests" / "reference.json").read_text(
        encoding="utf-8"
    )


class FakeClient:
    async def chat(self, request, **kwargs):
        return ChatResponse("Final answer: AC", None, "fake", {}, None, 0.0, {}, "stop")

    async def preflight(self) -> None:
        pass


@pytest.mark.asyncio
async def test_static_trial(tmp_path: Path) -> None:
    result = await run_trial(
        _problem(),
        0,
        tmp_path / "trial",
        FakeClient(),
        Sampling("teacher", max_tokens=128, temperature=0.0),
        client_name="vllm",
    )

    assert result["status"] == "passed"
    assert (tmp_path / "trial" / "agent" / "response.json").exists()
