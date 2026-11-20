import json
from dataclasses import replace
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.mmmupro.data.schemas import AnswerSpec, Problem, ProblemSpec
from ddsr_bench.benchmarks.mmmupro.evaluation.evaluator import evaluate, parse_answer
from ddsr_bench.benchmarks.mmmupro.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.mmmupro.evaluation.static import run_trial
from ddsr_bench.benchmarks.mmmupro.generation.prompts import SYSTEM, messages, render
from ddsr_bench.generation.client import ChatResponse, Sampling

IMAGE_1 = "data:image/png;base64,aW1hZ2Ux"
IMAGE_2 = "data:image/png;base64,aW1hZ2Uy"


def _problem(setting="standard10") -> Problem:
    choices = tuple(f"choice {index}" for index in range(10))
    choices = ("<image 1> option", *choices[1:])
    return Problem(
        ProblemSpec(
            f"{setting}/validation_Physics_1",
            "validation_Physics_1",
            setting,
            "Physics",
            "Science",
            "Compare <image 2>.",
            choices,
            (IMAGE_1,) if setting == "vision" else (IMAGE_1, IMAGE_2),
        ),
        AnswerSpec("C", "SECRET_EXPLANATION"),
    )


def test_standard_prompt_and_image_order() -> None:
    problem = _problem()
    text, images = render(problem.spec, "cot")

    assert text.startswith("Compare <image>.\nA. <image> option\nB. choice 1")
    assert text.endswith("Think step by step before answering.")
    assert images == (IMAGE_2, IMAGE_1)
    request = messages(problem.spec, "cot")
    assert request[0] == {"role": "system", "content": SYSTEM}
    assert request[1]["content"][0] == {"type": "text", "text": text}


def test_vision_prompt() -> None:
    problem = _problem("vision")
    text, images = render(problem.spec, "direct")

    assert text.startswith("Answer with the option letter")
    assert images == (IMAGE_1,)


def test_variable_option_count() -> None:
    problem = _problem()
    spec = replace(
        problem.spec, choices=problem.spec.choices + ("choice 10", "choice 11")
    )

    text, _ = render(spec, "direct")

    assert "L. choice 11" in text


def test_official_parser_and_score() -> None:
    problem = _problem()

    assert parse_answer("Reasoning.\nAnswer: C", problem.spec) == (
        "C",
        "answer-marker",
    )
    assert parse_answer("The result is (C).", problem.spec) == ("C", "bracket")
    assert evaluate("Answer: C", problem.spec, problem.answer)["reward"] == 1.0


def test_prepared_task_keeps_reference_private(tmp_path: Path) -> None:
    task = compile_problem(_problem(), tmp_path)
    public = (task / "instruction.md").read_text(encoding="utf-8")
    reference = (task / "tests" / "reference.json").read_text(encoding="utf-8")

    assert "SECRET" not in public
    assert "SECRET" in reference


@pytest.mark.asyncio
async def test_static_trial(tmp_path: Path) -> None:
    result = await run_trial(
        _problem(),
        "cot",
        0,
        tmp_path / "trial",
        FakeClient(),
        Sampling("teacher", max_tokens=128, temperature=0.0),
        client_name="vllm",
    )

    assert result["status"] == "passed"
    record = json.loads((tmp_path / "trial" / "agent" / "response.json").read_text())
    assert record["problem"]["setting"] == "standard10"


class FakeClient:
    async def chat(self, request, **kwargs):
        return ChatResponse("Answer: C", None, "fake", {}, None, 0.0, {}, "stop")

    async def preflight(self) -> None:
        pass
