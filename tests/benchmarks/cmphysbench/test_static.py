import json
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.cmphysbench.data.schemas import (
    AnswerSpec,
    Problem,
    ProblemSpec,
)
from ddsr_bench.benchmarks.cmphysbench.evaluation import static
from ddsr_bench.benchmarks.cmphysbench.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.cmphysbench.evaluation.static import run_job, run_trial
from ddsr_bench.benchmarks.utils import Resources
from ddsr_bench.generation.client import ChatResponse, Sampling

ANSWER = AnswerSpec("x^2/y")


def problem(answer: AnswerSpec | None = ANSWER, problem_id: str = "7") -> Problem:
    return Problem(
        ProblemSpec(
            id=problem_id,
            context="",
            question="Find the ratio.",
            symbols="$x$, $y$",
            answer_type="Expression",
            topic="Theoretical Foundations",
        ),
        answer,
    )


class FakeClient:
    def __init__(self) -> None:
        self.preflights = 0

    async def preflight(self) -> None:
        self.preflights += 1

    async def chat(self, messages: tuple[dict[str, str], ...]) -> ChatResponse:
        return ChatResponse(
            content=r"<think>derive</think>\boxed{\frac{x^{2}}{y}}",
            reasoning=None,
            model="teacher",
            usage={"completion_tokens": 8},
            seed=3,
            latency=0.1,
            raw={"id": "response-1"},
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_trial(tmp_path: Path) -> None:
    directory = tmp_path / "7__attempt-0"
    sampling = Sampling("teacher", max_tokens=16_384)

    result = await run_trial(
        problem(),
        directory=directory,
        attempt=0,
        client=FakeClient(),
        sampling=sampling,
        client_name="vllm",
    )

    assert result["reward"] == 1
    assert (directory / "artifacts" / "answer.txt").read_text() == (
        r"<think>derive</think>\boxed{\frac{x^{2}}{y}}"
    )
    response = (directory / "agent" / "response.json").read_text()
    assert "x^2/y" not in response
    trial = json.loads((directory / "result.json").read_text())
    assert trial["task_name"] == "cmphysbench/7"
    assert trial["attempt"] == 0


@pytest.mark.asyncio
async def test_reference_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no reference"):
        await run_trial(
            problem(None),
            0,
            tmp_path / "trial",
            FakeClient(),
            Sampling("teacher"),
            client_name="vllm",
        )

    assert not (tmp_path / "trial").exists()


@pytest.mark.asyncio
async def test_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    async def fake_trial(
        problem: Problem, attempt: int, directory: Path, *args, **kwargs
    ):
        calls.append((problem.spec.id, attempt, directory.name))
        return {"reward": 1.0, "status": "passed"}

    compile_problem(
        problem(problem_id="8"),
        tmp_path / "tasks" / "cmphysbench",
        Resources("cmphysbench:test", 2, 4096, 120),
    )

    def unavailable(**kwargs):
        raise ConnectionError

    monkeypatch.setattr(static, "load_split", unavailable)
    monkeypatch.setattr(static, "run_trial", fake_trial)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "job.yaml"
    config.write_text(
        """benchmark: cmphysbench
job_name: batch
jobs_dir: outputs/harbor
n_attempts: 2
n_concurrent_trials: 2
verifier:
  override_timeout_sec: 120
agents:
  - name: ddsr_bench.benchmarks.cmphysbench.evaluation.harbor:CMPhysBenchAgent
    model_name: teacher
    kwargs:
      client_name: vllm
      sampling: {max_tokens: 16384, temperature: 0.6, top_p: 0.95}
datasets:
  - path: tasks/cmphysbench
""",
        encoding="utf-8",
    )
    client = FakeClient()

    result = await run_job(config, client, task_name="8")

    assert result == {
        "output": "outputs/static/batch",
        "source": "prepared",
        "trials": 2,
        "scored": 2,
        "passed": 2,
        "errors": 0,
    }
    assert [call[0] for call in calls] == ["8", "8"]
    assert client.preflights == 1
