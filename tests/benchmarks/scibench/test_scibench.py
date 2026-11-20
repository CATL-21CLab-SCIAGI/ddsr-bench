import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.scibench.data.loader import load_problem
from ddsr_bench.benchmarks.scibench.data.schemas import AnswerSpec, Problem, ProblemSpec
from ddsr_bench.benchmarks.scibench.evaluation.evaluator import evaluate
from ddsr_bench.benchmarks.scibench.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.scibench.evaluation.static import run_trial
from ddsr_bench.benchmarks.scibench.generation.prompts import (
    direct_messages,
    display_unit,
    system_prompt,
)
from ddsr_bench.benchmarks.scibench.generation.runner import generate
from ddsr_bench.generation.client import ChatResponse, Sampling


def _record() -> dict[str, str]:
    return {
        "problem_text": "Compute the value.",
        "answer_latex": "100",
        "answer_number": "100",
        "unit": r"$10^{-2} \\mathrm{m}$",
        "source": "fund",
        "problemid": "1.2",
        "comment": "",
    }


def _problem() -> Problem:
    return load_problem(_record(), split="text", textbook="fund", index=0)


def test_loader_separates_reference() -> None:
    problem = _problem()

    assert problem.spec.id == "text/fund/000"
    assert problem.answer == AnswerSpec("100", r"$10^{-2} \\mathrm{m}$")
    assert "answer_number" not in json.dumps(asdict(problem.spec))


def test_official_prompt_and_unit_rewrite() -> None:
    problem = _problem()

    assert system_prompt().startswith("\nPlease provide")
    assert system_prompt().endswith("\n")
    assert display_unit(problem.spec.unit) == r"\\mathrm{m}$"
    assert direct_messages(problem.spec)[1]["content"].endswith("\nA: The answer is")


def test_official_numeric_score() -> None:
    problem = _problem()
    plain = ProblemSpec(
        "text/fund/1",
        "text",
        "fund",
        "1",
        "Question",
        "m",
        None,
        (),
    )

    assert (
        evaluate(r"The answer is \\boxed{105}", plain, AnswerSpec("100", "m"))["reward"]
        == 1.0
    )
    assert (
        evaluate(r"The answer is \\boxed{106}", plain, AnswerSpec("100", "m"))["reward"]
        == 0.0
    )
    assert (
        evaluate(r"The answer is \\boxed{1}", problem.spec, problem.answer)["reward"]
        == 1.0
    )


@pytest.mark.asyncio
async def test_zero_cot_uses_two_independent_calls() -> None:
    client = FakeClient()
    answer, record = await generate(
        _problem().spec,
        "zero-cot",
        client,
        Sampling("teacher", max_tokens=128),
    )

    assert answer == r"\\boxed{1}"
    assert len(client.requests) == len(record["calls"]) == 2
    assert [message["role"] for message in client.requests[1]] == ["system", "user"]
    assert "derivationTherefore" in client.requests[1][1]["content"]


def test_prepared_task_keeps_reference_private(tmp_path: Path) -> None:
    task = compile_problem(_problem(), tmp_path)

    assert '"number"' not in (task / "instruction.md").read_text()
    assert '"number": "100"' in (task / "tests" / "reference.json").read_text()


@pytest.mark.asyncio
async def test_static_trial(tmp_path: Path) -> None:
    result = await run_trial(
        _problem(),
        "zero-cot",
        0,
        tmp_path / "trial",
        FakeClient(),
        Sampling("teacher", max_tokens=128),
        client_name="vllm",
    )

    assert result["status"] == "passed"


class FakeClient:
    def __init__(self) -> None:
        self.requests = []

    async def preflight(self) -> None:
        pass

    async def chat(self, request, **kwargs):
        self.requests.append(request)
        content = "derivation" if len(self.requests) == 1 else r"\\boxed{1}"
        return ChatResponse(content, None, "fake", {}, None, 0.0, {}, "stop")
