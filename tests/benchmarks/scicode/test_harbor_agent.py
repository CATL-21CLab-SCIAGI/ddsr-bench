import json
from pathlib import Path

import pytest
from harbor.models.agent.context import AgentContext

from ddsr_bench.benchmarks.scicode.harbor import SciCodeAgent
from ddsr_bench.benchmarks.scicode.loader import load_problem
from ddsr_bench.benchmarks.scicode.prepare import encode_instruction
from ddsr_bench.generation.client import ChatMessages, ChatResponse, Sampling

FIXTURE = Path(__file__).parents[2] / "fixtures" / "scicode_19.json"


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[ChatMessages] = []
        self.ready = False

    async def preflight(self) -> None:
        self.ready = True

    async def chat(self, messages: ChatMessages) -> ChatResponse:
        self.calls.append(messages)
        number = len(self.calls)
        content = f"```python\ndef step_{number}():\n    return {number}\n```"
        return ChatResponse(
            content,
            None,
            "teacher",
            {"prompt_tokens": 10, "completion_tokens": 5},
            7,
            0.1,
            {"response": number},
            "stop",
        )


class FakeEnvironment:
    solution = ""

    async def exec(self, command: str, *, user: str) -> None:
        assert (command, user) == ("mkdir -p /app", "root")

    async def upload_file(self, source_path: str, target_path: str) -> None:
        assert target_path == "/app/solution.py"
        self.solution = Path(source_path).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_generates_cumulative_solution(tmp_path: Path) -> None:
    problem = load_problem(json.loads(FIXTURE.read_text(encoding="utf-8")))
    client = FakeClient()
    environment = FakeEnvironment()
    context = AgentContext()
    agent = SciCodeAgent(
        tmp_path, "teacher", client=client, sampling=Sampling("teacher", seed=7)
    )

    await agent.run(encode_instruction(problem), environment, context)

    record = json.loads((tmp_path / "response.json").read_text(encoding="utf-8"))
    assert client.ready
    assert len(client.calls) == 2
    assert "def step_1" in environment.solution
    assert "def step_2" in environment.solution
    assert problem.dependencies in environment.solution
    assert "assert np.allclose" not in json.dumps(record)
    assert [step["id"] for step in record["steps"]] == ["19.1", "19.2"]
    assert context.n_input_tokens == 20
    assert context.n_output_tokens == 10
