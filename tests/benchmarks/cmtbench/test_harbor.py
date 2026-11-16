import json
from pathlib import Path

import pytest
from harbor.models.agent.context import AgentContext

from ddsr_bench.benchmarks.cmtbench.data.schemas import ProblemSpec
from ddsr_bench.benchmarks.cmtbench.evaluation.harbor import CMTBenchAgent
from ddsr_bench.benchmarks.cmtbench.evaluation.prepare import encode_instruction
from ddsr_bench.generation.client import ChatMessages, ChatResponse, Sampling


class FakeClient:
    ready = False

    async def preflight(self) -> None:
        self.ready = True

    async def chat(self, messages: ChatMessages) -> ChatResponse:
        return ChatResponse(
            r"derivation \boxed{E}",
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
    problem = ProblemSpec("39", "Find E.", "Other")
    client = FakeClient()
    environment = FakeEnvironment()
    context = AgentContext()
    agent = CMTBenchAgent(
        tmp_path, "teacher", client=client, sampling=Sampling("teacher", seed=7)
    )

    await agent.run(encode_instruction(problem), environment, context)

    record = json.loads((tmp_path / "response.json").read_text(encoding="utf-8"))
    assert client.ready
    assert environment.answer == r"derivation \boxed{E}"
    assert record["problem_id"] == "39"
    assert set(record["problem"]) == {"prompt", "type"}
    assert context.n_input_tokens == 10
    assert context.n_output_tokens == 5
