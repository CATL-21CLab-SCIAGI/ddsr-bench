import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.mmlu.data.schemas import (
    AnswerSpec,
    Example,
    Problem,
    ProblemSpec,
)
from ddsr_bench.benchmarks.mmlu.evaluation.evaluator import evaluate, parse_answer
from ddsr_bench.benchmarks.mmlu.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.mmlu.evaluation.static import run_trial
from ddsr_bench.benchmarks.mmlu.generation.prompts import messages, render
from ddsr_bench.generation.client import ChatResponse, Sampling
from tests.benchmarks.helpers import assert_static_pipeline


def _problem() -> Problem:
    examples = tuple(
        Example(f"Example {index}?", ("one", "two", "three", "four"), "ABCD"[index % 4])
        for index in range(5)
    )
    return Problem(
        ProblemSpec(
            "abstract_algebra-0",
            "abstract_algebra",
            "STEM",
            "Target?",
            ("alpha", "beta", "gamma", "delta"),
            examples,
        ),
        AnswerSpec("C"),
    )


def test_prompt_matches_original_completion_format() -> None:
    problem = _problem()
    prompt = render(problem.spec)

    assert prompt.startswith(
        "The following are multiple choice questions (with answers) about  abstract algebra.\n\n"
    )
    assert "Example 0?\nA. one\nB. two\nC. three\nD. four\nAnswer: A\n\n" in prompt
    assert prompt.endswith("Target?\nA. alpha\nB. beta\nC. gamma\nD. delta\nAnswer:")
    assert messages(problem.spec) == ({"role": "user", "content": prompt},)


def test_generated_answer_parser_and_score() -> None:
    problem = _problem()

    assert parse_answer(" C") == "C"
    assert parse_answer("Reasoning\nAnswer: (C)") == "C"
    assert parse_answer("no final choice") is None
    assert evaluate("C", problem.spec, problem.answer)["reward"] == 1.0


def test_prepared_task_keeps_target_private(tmp_path: Path) -> None:
    problem = _problem()
    task = compile_problem(problem, tmp_path)
    public = json.loads((task / "instruction.md").read_text(encoding="utf-8"))
    reference = json.loads(
        (task / "tests" / "reference.json").read_text(encoding="utf-8")
    )

    assert public == json.loads(json.dumps(asdict(problem.spec)))
    assert reference == {"letter": "C"}
    assert "letter" not in public


@pytest.mark.asyncio
async def test_static_trial(tmp_path: Path) -> None:
    problem = _problem()
    client = FakeClient()

    result = await run_trial(
        problem,
        0,
        tmp_path / "trial",
        client,
        Sampling("teacher", max_tokens=64, temperature=0.0),
        client_name="vllm",
    )

    assert result["status"] == "passed"
    assert (tmp_path / "trial" / "artifacts" / "answer.txt").read_text() == "C"
    assert_static_pipeline(tmp_path / "trial", "mmlu")


class FakeClient:
    async def chat(self, request, **kwargs):
        return ChatResponse("C", None, "fake", {}, None, 0.0, {}, "stop")

    async def preflight(self) -> None:
        pass
