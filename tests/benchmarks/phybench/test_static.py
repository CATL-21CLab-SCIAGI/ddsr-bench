import json
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.phybench.data.loader import load_problem
from ddsr_bench.benchmarks.phybench.data.schemas import Problem
from ddsr_bench.benchmarks.phybench.evaluation import static
from ddsr_bench.benchmarks.phybench.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.phybench.evaluation.static import run_job, run_trial
from ddsr_bench.benchmarks.utils import Resources
from ddsr_bench.generation.client import ChatResponse, Sampling

FIXTURE = Path(__file__).parents[2] / "fixtures" / "phybench_133.json"


def problem() -> Problem:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return load_problem(record)


class FakeClient:
    def __init__(self) -> None:
        self.preflights = 0

    async def preflight(self) -> None:
        self.preflights += 1

    async def chat(self, messages: tuple[dict[str, str], ...]) -> ChatResponse:
        return ChatResponse(
            content=r"Derivation. \boxed{\frac{v^3}{R v_x}}",
            reasoning=None,
            model="teacher",
            usage={"completion_tokens": 12},
            seed=3,
            latency=0.1,
            raw={"id": "response-1"},
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_trial(tmp_path: Path) -> None:
    directory = tmp_path / "133__attempt-0"

    result = await run_trial(
        problem(),
        attempt=0,
        directory=directory,
        client=FakeClient(),
        sampling=Sampling("teacher", max_tokens=32_768),
        client_name="vllm",
    )

    assert result["reward"] == 1
    assert result["tag"] == "MECHANICS"
    assert (directory / "artifacts" / "answer.txt").read_text() == (
        r"Derivation. \boxed{\frac{v^3}{R v_x}}"
    )
    response = (directory / "agent" / "response.json").read_text()
    assert "natural coordinate system" not in response
    trial = json.loads((directory / "result.json").read_text())
    assert trial["task_name"] == "phybench/133"
    assert trial["attempt"] == 0

    class NoCalls:
        async def chat(self, *args, **kwargs):
            pytest.fail("completed trial must not call the model")

    assert (
        await run_trial(
            problem(),
            0,
            directory,
            NoCalls(),
            Sampling("teacher"),
            client_name="vllm",
            resume=True,
        )
        == result
    )


@pytest.mark.asyncio
async def test_reference_required(tmp_path: Path) -> None:
    source = problem()

    with pytest.raises(ValueError, match="has no reference"):
        await run_trial(
            Problem(source.spec, None),
            0,
            tmp_path / "trial",
            FakeClient(),
            Sampling("teacher"),
            client_name="vllm",
        )

    assert not (tmp_path / "trial").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_error", [OSError, ValueError])
async def test_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_error: type[Exception]
) -> None:
    calls = []

    def unavailable(**kwargs):
        raise source_error("source unavailable")

    async def fake_trial(
        problem: Problem, attempt: int, directory: Path, *args, **kwargs
    ):
        calls.append((problem.spec.id, attempt, directory.name))
        return {"reward": 1.0, "status": "passed"}

    compile_problem(
        problem(),
        tmp_path / "tasks" / "phybench",
        Resources("phybench:test", 2, 4096, 120),
    )
    monkeypatch.setattr(static, "load_split", unavailable)
    monkeypatch.setattr(static, "run_trial", fake_trial)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "job.yaml"
    config.write_text(
        """benchmark: phybench
job_name: batch
jobs_dir: outputs/harbor
n_attempts: 2
n_concurrent_trials: 2
agents:
  - name: ddsr_bench.benchmarks.phybench.evaluation.harbor:PHYBenchAgent
    model_name: teacher
    kwargs:
      client_name: vllm
      sampling: {max_tokens: 32768, temperature: 0.6, top_p: 0.95}
datasets:
  - path: tasks/phybench
""",
        encoding="utf-8",
    )
    client = FakeClient()

    result = await run_job(config, client, task_name="133")

    assert result == {
        "output": "outputs/static/batch",
        "source": "prepared",
        "trials": 2,
        "scored": 2,
        "passed": 2,
        "errors": 0,
    }
    assert calls == [
        ("133", 0, "133__attempt-0"),
        ("133", 1, "133__attempt-1"),
    ]
    assert client.preflights == 1
