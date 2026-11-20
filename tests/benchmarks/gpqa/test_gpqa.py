import json
import random
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.gpqa.data.loader import load_problem
from ddsr_bench.benchmarks.gpqa.evaluation.evaluator import evaluate, parse_answer
from ddsr_bench.benchmarks.gpqa.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.gpqa.evaluation.static import run_trial
from ddsr_bench.benchmarks.gpqa.generation.prompts import SYSTEM, messages, render
from ddsr_bench.generation.client import ChatResponse, Sampling


def _problem():
    record = {
        "Record ID": "rec-1",
        "Question": "Which choice is correct?",
        "Correct Answer": "correct",
        "Incorrect Answer 1": "wrong one",
        "Incorrect Answer 2": "wrong two",
        "Incorrect Answer 3": "wrong three",
        "Explanation": "SECRET_EXPLANATION",
        "High-level domain": "Physics",
        "Subdomain": "Mechanics",
    }
    return load_problem(record, "main", random.Random(0))


class FakeClient:
    async def chat(self, request, **kwargs):
        return ChatResponse(
            "The correct answer is (B)", None, "fake", {}, None, 0.0, {}, "stop"
        )

    async def preflight(self) -> None:
        pass


def test_projection_and_prompt_parity() -> None:
    problem = _problem()
    public = json.dumps(asdict(problem.spec))
    expected = (
        f"What is the correct answer to this question: {problem.spec.question}\n\n"
        "Choices:\n"
        + "\n".join(
            f"({letter}) {choice}"
            for letter, choice in zip("ABCD", problem.spec.choices, strict=True)
        )
        + '\n\nFormat your response as follows: "The correct answer is '
        '(insert answer here)"'
    )

    assert "SECRET" not in public
    assert render(problem.spec) == expected
    assert messages(problem.spec) == (
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": expected},
    )


def test_official_parser_and_score() -> None:
    problem = _problem()
    response = f"Reasoning. The correct answer is ({problem.answer.letter})"

    assert parse_answer(response) == problem.answer.letter
    assert evaluate(response, problem.spec, problem.answer)["reward"] == 1.0
    assert parse_answer("no final option") is None


def test_prepared_task_keeps_reference_private(tmp_path: Path) -> None:
    problem = _problem()
    task = compile_problem(problem, tmp_path)

    assert "SECRET" not in (task / "instruction.md").read_text(encoding="utf-8")
    assert "SECRET" in (task / "tests" / "reference.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_static_trial(tmp_path: Path) -> None:
    problem = _problem()
    answer = f"The correct answer is ({problem.answer.letter})"
    client = FakeClient()
    client.chat = lambda request, **kwargs: _response(answer)

    result = await run_trial(
        problem,
        0,
        tmp_path / "trial",
        client,
        Sampling("teacher", max_tokens=128, temperature=0.0),
        client_name="vllm",
    )

    record = (tmp_path / "trial" / "agent" / "response.json").read_text(
        encoding="utf-8"
    )
    assert result["status"] == "passed"
    assert "SECRET" not in record


async def _response(content: str) -> ChatResponse:
    return ChatResponse(content, None, "fake", {}, None, 0.0, {}, "stop")
