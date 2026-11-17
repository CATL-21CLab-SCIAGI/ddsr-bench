import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.prismphysics.data.loader import load_problem
from ddsr_bench.benchmarks.prismphysics.generation.prompts import messages, prompt
from ddsr_bench.benchmarks.prismphysics.generation.runner import generate
from ddsr_bench.generation.client import ChatResponse, Sampling

FIXTURE = Path(__file__).parents[2] / "fixtures" / "prismphysics_02_3006.json"


def problem():
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return load_problem(value, "02")


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    async def chat(self, request):
        self.calls.append(request)
        return ChatResponse(
            content=r"Derivation. $$C=\frac{\varepsilon S}{d}$$",
            reasoning="teacher reasoning",
            model="teacher",
            usage={"completion_tokens": 12},
            seed=3,
            latency=0.1,
            raw={"id": "response-1"},
            finish_reason="stop",
        )


def test_upstream_prompt_parity() -> None:
    rendered = prompt(problem().spec)

    assert sha256(rendered.encode()).hexdigest() == (
        "23cfc5da74245f35088e5f724b75e2286839be444363dfbb4a7ae74f17717a91"
    )
    assert "\f" in rendered
    assert "Your final answer should be given" not in rendered
    assert messages(problem().spec) == ({"role": "user", "content": rendered},)


@pytest.mark.asyncio
async def test_single_call() -> None:
    source = problem()
    client = FakeClient()
    sampling = Sampling("teacher", max_tokens=8_096)

    content, record = await generate(source.spec, client, sampling)

    assert client.calls == [messages(source.spec)]
    assert content.startswith("Derivation")
    assert record["sampling"] == asdict(sampling)
    assert record["responses"][0]["reasoning"] == "teacher reasoning"
    assert record["problem"]["domain"] == "electromagnetism"
    assert record["problem"]["images"] == []
    serialized = json.dumps(record)
    assert "grading_standard" not in serialized
    assert "parallel-plate capacitor" not in serialized
