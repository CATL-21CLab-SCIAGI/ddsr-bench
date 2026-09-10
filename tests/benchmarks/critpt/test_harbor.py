import json
from pathlib import Path

import pytest
from harbor.models.agent.context import AgentContext

from ddsr_bench.benchmarks.critpt.harbor import CritPtAgent
from ddsr_bench.benchmarks.critpt.prepare import encode_instruction
from ddsr_bench.benchmarks.critpt.schemas import ProblemSpec
from ddsr_bench.generation.client import ChatMessages, ChatResponse, Sampling


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[ChatMessages] = []
        self.ready = False

    async def preflight(self) -> None:
        self.ready = True

    async def chat(self, messages: ChatMessages) -> ChatResponse:
        self.calls.append(messages)
        content = (
            "reasoned answer"
            if len(self.calls) == 1
            else "```python\ndef answer():\n    return 42\n```"
        )
        return ChatResponse(
            content=content,
            reasoning="reasoning",
            model="solver",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            seed=7,
            latency=0.1,
            raw={"choices": [{"message": {"content": content}}]},
        )


class FakeEnvironment:
    def __init__(self) -> None:
        self.answer = ""
        self.target = ""

    async def exec(self, command: str, *, user: str) -> None:
        assert command == "mkdir -p /app"
        assert user == "root"

    async def upload_file(self, source_path: str, target_path: str) -> None:
        self.answer = Path(source_path).read_text(encoding="utf-8")
        self.target = target_path


def test_harbor_configuration(tmp_path: Path) -> None:
    agent = CritPtAgent(
        tmp_path,
        "solver",
        client_name="bedrock",
        sampling={"max_tokens": 123, "reasoning_effort": "low"},
        stream=True,
    )

    assert agent.sampling == Sampling("solver", max_tokens=123, reasoning_effort="low")
    assert agent.stream is True
    assert agent.api_key_env == "AWS_BEARER_TOKEN_BEDROCK"


@pytest.mark.asyncio
async def test_harbor_artifacts(tmp_path: Path) -> None:
    problem = ProblemSpec(
        id="p1",
        type="main",
        index=None,
        statement="Find the result.",
        code_template="def answer():\n    return ...",
        source="critpt",
        source_path=Path("private/location.json"),
    )
    client = FakeClient()
    environment = FakeEnvironment()
    context = AgentContext()
    agent = CritPtAgent(
        tmp_path,
        "solver",
        style="two-step",
        sampling=Sampling("solver", seed=7),
        client=client,
    )

    await agent.run(encode_instruction(problem), environment, context)

    record = json.loads((tmp_path / "response.json").read_text(encoding="utf-8"))
    assert client.ready
    assert len(client.calls) == 2
    assert environment.target == "/app/answer.py"
    assert environment.answer == "def answer():\n    return 42\n"
    assert record["problem_id"] == "p1"
    assert record["strategy"] == "two-step"
    assert record["seed"] == 7
    assert len(record["messages"]) == 5
    assert len(record["responses"]) == 2
    assert record["responses"][0]["reasoning"] == "reasoning"
    assert "private/location.json" not in json.dumps(record)
    assert context.n_input_tokens == 20
    assert context.n_output_tokens == 10
