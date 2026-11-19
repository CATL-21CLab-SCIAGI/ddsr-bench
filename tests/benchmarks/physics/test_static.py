import json
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.physics.data.loader import load_problem
from ddsr_bench.benchmarks.physics.evaluation import static
from ddsr_bench.benchmarks.physics.evaluation.prepare import compile_problem
from ddsr_bench.generation.client import ChatResponse, Sampling

FIXTURE = Path(__file__).parents[2] / "fixtures" / "physics_atomic_1_1.json"


def _problem():
    return load_problem(json.loads(FIXTURE.read_text(encoding="utf-8")), "test")


class FakeClient:
    def __init__(self, content: str):
        self.content = content

    async def preflight(self) -> None:
        pass

    async def chat(self, messages, **kwargs) -> ChatResponse:
        return ChatResponse(
            self.content,
            None,
            "fake",
            {"prompt_tokens": 3, "completion_tokens": 2},
            7,
            0.1,
            {},
            "stop",
        )


class FailingClient(FakeClient):
    async def chat(self, messages, **kwargs) -> ChatResponse:
        raise RuntimeError("generation failed")


@pytest.mark.asyncio
async def test_trial_keeps_reference_private(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    result = await static.run_trial(
        _problem(),
        0,
        trial,
        FakeClient("Explanation: yes\nAnswers: 1. yes\nConfidence: 90%"),
        FakeClient(json.dumps({"answers": [{"correct": "yes"}], "confidence": 90})),
        Sampling("teacher", seed=7),
        client_name="vllm",
        judge_name="vllm",
        judge_model="judge",
        judge_max_tokens=32768,
        judge_retries=1,
    )

    record = (trial / "agent" / "response.json").read_text(encoding="utf-8")
    assert result["reward"] == 1
    assert result["judge_model"] == "judge"
    assert "800,000" not in record
    assert "final_answers" not in record


@pytest.mark.asyncio
async def test_generation_failure_is_scored_as_unanswered(tmp_path: Path) -> None:
    result = await static.run_trial(
        _problem(),
        0,
        tmp_path / "trial",
        FailingClient("unused"),
        FakeClient("unused"),
        Sampling("teacher"),
        client_name="vllm",
        judge_name="vllm",
        judge_model="judge",
        judge_max_tokens=32768,
        judge_retries=1,
    )

    assert result["status"] == "error"
    assert result["judged"] is True
    assert result["judgment"]["answers"][0]["correct"] is False


@pytest.mark.asyncio
async def test_prepared_job_fallback(tmp_path: Path, monkeypatch) -> None:
    tasks = tmp_path / "tasks"
    compile_problem(_problem(), tasks)
    config = tmp_path / "job.yaml"
    config.write_text(
        f"""benchmark: physics
job_name: demo
jobs_dir: {tmp_path / 'harbor'}
n_attempts: 1
n_concurrent_trials: 1
agents:
  - name: physics
    model_name: teacher
    kwargs:
      client_name: vllm
      sampling: {{max_tokens: 100}}
      judge: {{client_name: vllm, model: judge}}
datasets:
  - path: {tasks}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        static, "load_split", lambda *args: (_ for _ in ()).throw(ValueError())
    )

    summary = await static.run_job(
        config,
        FakeClient("Answers: yes"),
        FakeClient(json.dumps({"answers": [{"correct": "yes"}], "confidence": 80})),
    )

    assert summary["source"] == "prepared"
    assert summary["trials"] == 1
    assert summary["passed"] == 1
    assert Path(summary["output"]).is_dir()
