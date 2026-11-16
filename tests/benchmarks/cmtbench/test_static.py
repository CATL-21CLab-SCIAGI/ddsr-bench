import json
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.cmtbench.data.schemas import (
    AnswerSpec,
    Problem,
    ProblemSpec,
)
from ddsr_bench.benchmarks.cmtbench.evaluation import static
from ddsr_bench.benchmarks.cmtbench.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.cmtbench.evaluation.static import run_job, run_trial
from ddsr_bench.benchmarks.utils import Resources
from ddsr_bench.generation.client import ChatResponse, Sampling


def problem(answer: AnswerSpec | None = None) -> Problem:
    return Problem(
        ProblemSpec("39", "Find the expression.", "Other"),
        answer or AnswerSpec(r"\boxed{2x}", "$x$", ""),
    )


class FakeClient:
    def __init__(self) -> None:
        self.preflights = 0

    async def preflight(self) -> None:
        self.preflights += 1

    async def chat(self, messages: tuple[dict[str, str], ...]) -> ChatResponse:
        return ChatResponse(
            content=r"Derivation. \boxed{x+x}",
            reasoning=None,
            model="teacher",
            usage={"completion_tokens": 8},
            seed=3,
            latency=0.1,
            raw={"id": "response-1"},
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_trial_and_resume(tmp_path: Path) -> None:
    directory = tmp_path / "39__attempt-0"
    sampling = Sampling("teacher", max_tokens=8_192)
    result = await run_trial(
        problem(),
        0,
        directory,
        FakeClient(),
        sampling,
        client_name="vllm",
    )

    assert result["reward"] == 1
    assert result["type"] == "Other"
    assert (directory / "artifacts" / "answer.txt").read_text() == (
        r"Derivation. \boxed{x+x}"
    )
    response = (directory / "agent" / "response.json").read_text()
    assert r"\boxed{2x}" not in response
    trial = json.loads((directory / "result.json").read_text())
    assert (trial["task_name"], trial["attempt"]) == ("cmtbench/39", 0)

    class NoCalls:
        async def chat(self, *args, **kwargs):
            pytest.fail("completed trial must not call the model")

    resumed = await run_trial(
        problem(),
        0,
        directory,
        NoCalls(),
        sampling,
        client_name="vllm",
        resume=True,
    )
    assert resumed == result


@pytest.mark.asyncio
async def test_reference_required(tmp_path: Path) -> None:
    source = Problem(ProblemSpec("39", "Find it.", "Other"), None)

    with pytest.raises(ValueError, match="has no reference"):
        await run_trial(
            source,
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
        source: Problem, attempt: int, directory: Path, *args, **kwargs
    ):
        calls.append((source.spec.id, attempt, directory.name))
        return {"reward": 1.0, "status": "passed"}

    compile_problem(
        problem(),
        tmp_path / "tasks" / "cmtbench",
        Resources("cmtbench:test", 2, 4096, 120),
    )

    def unavailable(**kwargs):
        raise OSError("source unavailable")

    monkeypatch.setattr(static, "load_split", unavailable)
    monkeypatch.setattr(static, "run_trial", fake_trial)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "job.yaml"
    config.write_text(
        """benchmark: cmtbench
job_name: batch
jobs_dir: outputs/harbor
n_attempts: 2
n_concurrent_trials: 2
agents:
  - name: ddsr_bench.benchmarks.cmtbench.evaluation.harbor:CMTBenchAgent
    model_name: teacher
    kwargs:
      client_name: vllm
      sampling: {max_tokens: 8192, temperature: 0.0, top_p: 1.0}
datasets:
  - path: tasks/cmtbench
""",
        encoding="utf-8",
    )
    client = FakeClient()

    result = await run_job(config, client, task_name="39")

    assert result == {
        "output": "outputs/static/batch",
        "source": "prepared",
        "trials": 2,
        "scored": 2,
        "passed": 2,
        "errors": 0,
    }
    assert calls == [
        ("39", 0, "39__attempt-0"),
        ("39", 1, "39__attempt-1"),
    ]
    assert client.preflights == 1
