import json
from dataclasses import replace
from hashlib import sha256
from importlib import resources
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.hle.data.loader import load_problem
from ddsr_bench.benchmarks.hle.generation.prompts import (
    judge_prompt,
    messages,
    system_prompt,
)
from ddsr_bench.benchmarks.hle.generation.runner import generate
from ddsr_bench.generation.client import ChatResponse, Sampling

FIXTURE = Path(__file__).parents[2] / "fixtures" / "hle_gold.json"


def _problem():
    return load_problem(json.loads(FIXTURE.read_text(encoding="utf-8")))


class FakeClient:
    async def chat(self, request, **kwargs):
        self.request = request
        return ChatResponse("answer", None, "teacher", {}, 7, 0.1, {}, "stop")


def test_exact_prompts_and_o1_role() -> None:
    problem = _problem()
    root = resources.files("configs").joinpath("prompts", "hle")

    assert sha256(root.joinpath("system.txt").read_bytes()).hexdigest() == (
        "cd27d28bdfe308dab9c78659fb0ff24dee3d85cdf0340ec4e37f984c5fcf1aac"
    )
    assert sha256(root.joinpath("judge.txt").read_bytes()).hexdigest() == (
        "732dd73cab327aca3bc0d6357d0c2254c0fdeeb8928d2c51edc51648d1be4fb8"
    )
    assert system_prompt().startswith(
        "Your response should be in the following format:"
    )
    assert messages(problem.spec, "gpt-4o")[0]["role"] == "system"
    assert messages(problem.spec, "o1")[0]["role"] == "user"
    rendered = judge_prompt(problem.spec, "Answer: D", problem.answer)
    assert "[correct_answer]: D" in rendered
    assert r"0|\%| and 100|\%|" in rendered


def test_multimodal_message() -> None:
    problem = replace(_problem().spec, image="data:image/png;base64,AA==")

    request = messages(problem, "teacher")

    assert request[1]["content"][-1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AA=="},
    }


@pytest.mark.asyncio
async def test_generation_record_is_public() -> None:
    client = FakeClient()
    content, record = await generate(_problem().spec, client, Sampling("teacher"))

    assert content == "answer"
    assert record["strategy"] == "single-turn"
    assert len(record["messages"]) == 3
    serialized = json.dumps(record)
    assert '"answer": "D"' not in serialized
    assert "rationale" not in serialized
