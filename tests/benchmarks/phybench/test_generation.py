import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.phybench.data.loader import load_problem
from ddsr_bench.benchmarks.phybench.data.schemas import Problem
from ddsr_bench.benchmarks.phybench.generation.prompts import (
    messages,
    prompt_template,
    user_prompt,
)
from ddsr_bench.benchmarks.phybench.generation.runner import generate
from ddsr_bench.generation.client import ChatResponse, Sampling

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "fixtures" / "phybench_133.json"
SNAPSHOT = ROOT / "snapshots" / "phybench_prompt.txt"


def problem() -> Problem:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return load_problem(record)


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    async def preflight(self) -> None:
        pass

    async def chat(self, request: tuple[dict[str, str], ...]) -> ChatResponse:
        self.calls.append(request)
        return ChatResponse(
            content="Derivation. \\boxed{\\frac{v^3}{R v_x}}",
            reasoning="internal reasoning",
            model="teacher",
            usage={"completion_tokens": 20},
            seed=3,
            latency=0.1,
            raw={"id": "response-1"},
            finish_reason="stop",
        )


def test_official_prompt() -> None:
    expected = SNAPSHOT.read_text(encoding="utf-8").removesuffix("\n")

    assert sha256(prompt_template().encode()).hexdigest() == (
        "5bf88a0dea2da7a082e82a4616bce91b5ef0a72e95c8407403e9457d71a92f8a"
    )
    assert user_prompt(problem().spec) == expected
    assert messages(problem().spec) == ({"role": "user", "content": expected},)


@pytest.mark.asyncio
async def test_single_call() -> None:
    source = problem()
    client = FakeClient()
    sampling = Sampling("teacher", max_tokens=32_768, temperature=0.6, top_p=0.95)

    content, record = await generate(source.spec, client, sampling)

    assert len(client.calls) == 1
    assert client.calls[0] == messages(source.spec)
    assert content.endswith(r"\boxed{\frac{v^3}{R v_x}}")
    assert record["sampling"] == asdict(sampling)
    assert record["responses"][0]["reasoning"] == "internal reasoning"
    assert set(record["problem"]) == {"tag", "content"}
    assert "natural coordinate system" not in str(record)
