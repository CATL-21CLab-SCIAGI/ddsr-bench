from dataclasses import asdict
from hashlib import sha256

import pytest

from ddsr_bench.benchmarks.cmphysbench.data.schemas import ProblemSpec
from ddsr_bench.benchmarks.cmphysbench.generation.prompts import (
    messages,
    system_prompt,
    user_prompt,
)
from ddsr_bench.benchmarks.cmphysbench.generation.runner import generate
from ddsr_bench.generation.client import ChatResponse, Sampling


def problem() -> ProblemSpec:
    return ProblemSpec(
        id="7",
        context="Use the supplied convention. ",
        question="Find the energy.",
        symbols="$k$: wave vector",
        answer_type="Expression",
        topic="Theoretical Foundations",
    )


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    async def preflight(self) -> None:
        pass

    async def chat(self, request: tuple[dict[str, str], ...]) -> ChatResponse:
        self.calls.append(request)
        return ChatResponse(
            content="<think>derive</think>\\boxed{k^2}",
            reasoning=None,
            model="teacher",
            usage={"completion_tokens": 8},
            seed=3,
            latency=0.1,
            raw={"id": "response-1"},
            finish_reason="stop",
        )


def test_upstream_prompts() -> None:
    expected = (
        "Use the supplied convention. Find the energy.\n"
        "Here are the relevant symbols:\n$k$: wave vector"
    )

    assert user_prompt(problem()) == expected
    assert messages(problem())[1] == {"role": "user", "content": expected}
    assert sha256(system_prompt().encode()).hexdigest() == (
        "019d3c4df3f9918bc47e11af14ed935f7a0cd0b006db57cfd32ab03d96570acb"
    )


@pytest.mark.asyncio
async def test_single_call() -> None:
    client = FakeClient()
    sampling = Sampling("teacher", max_tokens=16_384, temperature=0.6, top_p=0.95)

    content, record = await generate(problem(), client, sampling)

    assert len(client.calls) == 1
    assert client.calls[0] == messages(problem())
    assert content == "<think>derive</think>\\boxed{k^2}"
    assert record["sampling"] == asdict(sampling)
    assert record["messages"][-1]["role"] == "assistant"
    assert "final_answer" not in str(record)
