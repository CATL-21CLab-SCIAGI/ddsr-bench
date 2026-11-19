import json
from hashlib import sha256
from importlib import resources

import pytest

from ddsr_bench.benchmarks.physics.data.schemas import (
    AnswerSpec,
    Figure,
    ProblemSpec,
)
from ddsr_bench.benchmarks.physics.generation.prompts import (
    judge_prompt,
    messages,
    system_prompt,
)
from ddsr_bench.benchmarks.physics.generation.runner import generate
from ddsr_bench.generation.client import ChatResponse, Sampling


class FakeClient:
    async def chat(self, request, **kwargs):
        assert request == messages(PROBLEM)
        assert kwargs["stream_path"] is None
        return ChatResponse(
            "Explanation: e\nAnswers: 1\nConfidence: 80%",
            "reasoning",
            "teacher",
            {"prompt_tokens": 3, "completion_tokens": 2},
            7,
            0.1,
            {"id": "response"},
            "stop",
        )


PROBLEM = ProblemSpec(
    "atomic/1-1",
    "validation",
    "Atomic Physics",
    "Question?",
    (Figure("data:image/png;base64,AA=="),),
)
ANSWER = AnswerSpec("SECRET_SOLUTION", ("x = 1",), ("1",), ("numeric",))


def test_exact_prompt_files() -> None:
    root = resources.files("configs").joinpath("prompts", "physics")

    assert sha256(root.joinpath("system.txt").read_bytes()).hexdigest() == (
        "17ffc9761eb3a67a475f4e2e036f9a5a0fe05d864a6b7e86c52c4589f89d738e"
    )
    assert sha256(root.joinpath("judge.txt").read_bytes()).hexdigest() == (
        "fb9a31a4242b0422056967125892ce694d2eb0030aa71bf935cfcf1bcea85c7f"
    )
    assert system_prompt().startswith("Your response should be")


def test_multimodal_messages() -> None:
    request = messages(PROBLEM)

    assert request[0] == {"role": "system", "content": system_prompt()}
    assert request[1]["content"] == [
        {"type": "text", "text": "Question?"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AA=="},
        },
    ]


def test_judge_prompt() -> None:
    prompt = judge_prompt(PROBLEM, "model response", ANSWER)

    assert "[question]: Question?" in prompt
    assert "[response]: model response" in prompt
    assert "0. label 1: x = 1" in prompt
    assert "1% relative difference" in prompt


def test_judge_values_are_not_reformatted() -> None:
    problem = ProblemSpec("p", "test", "Optics", "literal {references}", ())

    prompt = judge_prompt(problem, "literal {question}", ANSWER)

    assert "[question]: literal {references}" in prompt
    assert "[response]: literal {question}" in prompt


@pytest.mark.asyncio
async def test_generation_record_is_public() -> None:
    content, record = await generate(PROBLEM, FakeClient(), Sampling("teacher", seed=7))

    assert content.startswith("Explanation:")
    assert record["problem_id"] == "atomic/1-1"
    assert record["problem"]["figures"] == 1
    assert record["responses"][0]["reasoning"] == "reasoning"
    assert "SECRET" not in json.dumps(record)
