import json
from pathlib import Path

import pytest
from harbor.models.job.config import AgentConfig

from ddsr_bench.benchmarks.collect import collect_trials
from ddsr_bench.benchmarks.critpt.prepare import encode_instruction
from ddsr_bench.benchmarks.critpt.schemas import ProblemSpec
from ddsr_bench.benchmarks.critpt.static import run_job, run_trial
from ddsr_bench.generation.client import ChatMessages, ChatResponse


class FakeClient:
    def __init__(self) -> None:
        self.preflights = 0

    async def preflight(self) -> None:
        self.preflights += 1

    async def chat(self, messages: ChatMessages) -> ChatResponse:
        content = "```python\ndef answer():\n    return 42\n```"
        return ChatResponse(content, None, "local", None, None, 0.1, {})


class InvalidClient(FakeClient):
    async def chat(self, messages: ChatMessages) -> ChatResponse:
        return ChatResponse("not Python", None, "local", None, None, 0.1, {})


@pytest.mark.asyncio
async def test_valid_trial(tmp_path: Path) -> None:
    problem = ProblemSpec(
        id="p1",
        type="main",
        index=None,
        statement="Return 42.",
        code_template="def answer():\n    return ...\n",
        source="critpt",
        source_path=Path("private.json"),
    )
    output = tmp_path / "p1__attempt-0"
    agent = AgentConfig(
        name="ddsr_bench.benchmarks.critpt.harbor:CritPtAgent",
        model_name="local",
        kwargs={"client_name": "vllm", "style": "one-step"},
    )

    status = await run_trial(problem, 0, output, FakeClient(), agent)

    record = (output / "agent" / "response.json").read_text(encoding="utf-8")
    assert status == "validated"
    assert (output / "artifacts" / "answer.py").is_file()
    assert json.loads(record)["problem_id"] == "p1"
    assert "real_answer" not in record


@pytest.mark.asyncio
async def test_invalid_trial_has_no_reward(tmp_path: Path) -> None:
    problem = ProblemSpec(
        id="p1",
        type="main",
        index=None,
        statement="Return 42.",
        code_template="def answer():\n    return ...\n",
        source="critpt",
        source_path=Path("private.json"),
    )

    agent = AgentConfig(
        name="ddsr_bench.benchmarks.critpt.harbor:CritPtAgent", model_name="local"
    )
    output = tmp_path / "p1__attempt-0"

    status = await run_trial(problem, 0, output, InvalidClient(), agent)

    result = json.loads((output / "validation" / "result.json").read_text())
    assert status == "error"
    assert result["reward"] is None


@pytest.mark.asyncio
async def test_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    problem = ProblemSpec(
        "p1",
        "main",
        None,
        "Return 42.",
        "def answer():\n    return ...\n",
        "critpt",
        Path("source.json"),
    )
    task = tmp_path / "tasks" / "p1"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text(encode_instruction(problem))
    config = tmp_path / "job.yaml"
    config.write_text("""job_name: batch
jobs_dir: outputs/harbor
n_attempts: 2
n_concurrent_trials: 2
agents:
  - name: ddsr_bench.benchmarks.critpt.harbor:CritPtAgent
    model_name: local
    kwargs: {client_name: vllm, style: one-step}
datasets:
  - path: tasks
""")
    monkeypatch.chdir(tmp_path)
    client = FakeClient()

    result = await run_job(config, client)

    assert result == {
        "output": "outputs/static/batch",
        "trials": 2,
        "validated": 2,
        "errors": 0,
    }
    output = tmp_path / "outputs" / "static" / "batch"
    answers = output.glob("*/artifacts/answer.py")
    assert len(list(answers)) == 2
    assert client.preflights == 1

    summary = collect_trials(output)
    assert summary["complete_batches"] == 2
    assert summary["batches"][0]["validated"] == 1
    assert summary["batches"][0]["trials"][0]["reward"] is None
