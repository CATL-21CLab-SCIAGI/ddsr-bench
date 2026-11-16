import json
from pathlib import Path

import pytest
from harbor.models.agent.context import AgentContext

from ddsr_bench.benchmarks.prismphysics.data.loader import load_problem
from ddsr_bench.benchmarks.prismphysics.evaluation.harbor import PRISMPhysicsAgent
from ddsr_bench.benchmarks.prismphysics.evaluation.prepare import encode_instruction
from ddsr_bench.generation.client import ChatMessages, ChatResponse, Sampling

FIXTURE = Path(__file__).parents[2] / "fixtures" / "prismphysics_02_3006.json"


class FakeClient:
    ready = False

    async def preflight(self) -> None:
        self.ready = True

    async def chat(self, messages: ChatMessages) -> ChatResponse:
        return ChatResponse(
            r"derivation $$C=\varepsilon S/d$$",
            "reasoning",
            "teacher",
            {"prompt_tokens": 10, "completion_tokens": 5},
            7,
            0.1,
            {"response": 1},
            "stop",
        )


class FakeEnvironment:
    answer = ""

    async def exec(self, command: str, *, user: str) -> None:
        assert (command, user) == ("mkdir -p /app", "root")

    async def upload_file(self, source_path: str, target_path: str) -> None:
        assert target_path == "/app/answer.txt"
        self.answer = Path(source_path).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_harbor_artifact(tmp_path: Path) -> None:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problem = load_problem(value, "02")
    client = FakeClient()
    environment = FakeEnvironment()
    context = AgentContext()
    agent = PRISMPhysicsAgent(
        tmp_path, "teacher", client=client, sampling=Sampling("teacher", seed=7)
    )

    await agent.run(encode_instruction(problem.spec), environment, context)

    record = json.loads((tmp_path / "response.json").read_text(encoding="utf-8"))
    assert client.ready
    assert environment.answer == r"derivation $$C=\varepsilon S/d$$"
    assert record["problem_id"] == "02-3006"
    assert record["problem"]["domain"] == "electromagnetism"
    assert context.n_input_tokens == 10
    assert context.n_output_tokens == 5
