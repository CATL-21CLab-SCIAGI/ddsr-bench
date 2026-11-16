import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.cmtbench.data.loader import load_problem
from ddsr_bench.benchmarks.cmtbench.generation.prompts import messages, system_prompt
from ddsr_bench.benchmarks.cmtbench.generation.runner import generate
from ddsr_bench.generation.client import ChatResponse, Sampling

FIXTURE = Path(__file__).parents[2] / "fixtures" / "cmtbench_39.json"


def problem():
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return load_problem(value)


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    async def preflight(self) -> None:
        pass

    async def chat(self, request: tuple[dict[str, str], ...]) -> ChatResponse:
        self.calls.append(request)
        return ChatResponse(
            content=r"Derivation. \boxed{x}",
            reasoning="internal reasoning",
            model="teacher",
            usage={"completion_tokens": 12},
            seed=3,
            latency=0.1,
            raw={"id": "response-1"},
            finish_reason="stop",
        )


def test_official_messages() -> None:
    source = problem()
    request = messages(source.spec)

    assert sha256(system_prompt().encode()).hexdigest() == (
        "febe5e281a14f74bf4903e28c1360fefcf2d8c706e6ed1dd16eed307fed73dd2"
    )
    assert request == (
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": source.spec.prompt},
    )


@pytest.mark.asyncio
async def test_single_call() -> None:
    source = problem()
    client = FakeClient()
    sampling = Sampling("teacher", max_tokens=8_192)

    content, record = await generate(source.spec, client, sampling)

    assert client.calls == [messages(source.spec)]
    assert content == r"Derivation. \boxed{x}"
    assert record["sampling"] == asdict(sampling)
    assert record["responses"][0]["reasoning"] == "internal reasoning"
    assert set(record["problem"]) == {"prompt", "type"}
    assert all(
        field not in record["problem"]
        for field in ("solution", "parameters", "functions")
    )
