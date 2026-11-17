import json
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.prismphysics.data.loader import THRESHOLD, load_problem
from ddsr_bench.benchmarks.prismphysics.evaluation import static
from ddsr_bench.benchmarks.prismphysics.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.prismphysics.evaluation.static import run_job, run_trial
from ddsr_bench.benchmarks.utils import Resources
from ddsr_bench.generation.client import ChatResponse, Sampling

FIXTURE = Path(__file__).parents[2] / "fixtures" / "prismphysics_02_3006.json"


def problem():
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return load_problem(record, "02")


class FakeClient:
    def __init__(self) -> None:
        self.preflights = 0

    async def preflight(self) -> None:
        self.preflights += 1

    async def chat(self, messages) -> ChatResponse:
        return ChatResponse(
            content=r"$$C=\varepsilon_0L^2/d$$",
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
    directory = tmp_path / "02-3006__attempt-0"
    sampling = Sampling("teacher", max_tokens=8_096)
    result = await run_trial(
        problem(),
        0,
        directory,
        FakeClient(),
        sampling,
        client_name="vllm",
    )

    assert result["reward"] == 1
    assert result["domain"] == "electromagnetism"
    assert (directory / "artifacts" / "answer.txt").read_text() == (
        r"$$C=\varepsilon_0L^2/d$$"
    )
    response = (directory / "agent" / "response.json").read_text()
    assert r"\frac{\varepsilon_0" not in response
    trial = json.loads((directory / "result.json").read_text())
    assert (trial["task_name"], trial["attempt"]) == ("prismphysics/02-3006", 0)

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
async def test_job_from_prepared_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    async def fake_trial(source, attempt, directory, *args, **kwargs):
        calls.append((source.spec.id, attempt, directory.name))
        return {"reward": 0.5, "status": "different"}

    compile_problem(
        problem(),
        tmp_path / "tasks" / "prismphysics" / "official",
        Resources("prismphysics:test", 2, 4096, 120),
    )

    def unavailable(*args, **kwargs):
        raise OSError("source unavailable")

    monkeypatch.setattr(static, "load_split", unavailable)
    monkeypatch.setattr(static, "run_trial", fake_trial)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "job.yaml"
    config.write_text(
        """benchmark: prismphysics
job_name: batch
jobs_dir: outputs/harbor
n_attempts: 2
n_concurrent_trials: 2
agents:
  - name: ddsr_bench.benchmarks.prismphysics.evaluation.harbor:PRISMPhysicsAgent
    model_name: teacher
    kwargs:
      client_name: vllm
      sampling: {max_tokens: 8096, temperature: 0.0, top_p: 1.0}
datasets:
  - path: tasks/prismphysics/official
""",
        encoding="utf-8",
    )
    client = FakeClient()

    result = await run_job(config, client, task_name="02-3006")

    assert result == {
        "output": "outputs/static/batch",
        "source": "prepared",
        "trials": 2,
        "scored": 2,
        "passed": 0,
        "mean_score": 0.5,
        "errors": 0,
    }
    assert calls == [
        ("02-3006", 0, "02-3006__attempt-0"),
        ("02-3006", 1, "02-3006__attempt-1"),
    ]
    assert client.preflights == 1


def test_rejects_malformed_prepared_reference() -> None:
    with pytest.raises(TypeError, match="invalid prepared reference"):
        static._answer(
            {"formulas": [{"index": "one"}], "threshold": THRESHOLD},
            "02-3006",
            THRESHOLD,
        )
    with pytest.raises(ValueError, match="invalid formula DAG"):
        static._answer(
            {
                "formulas": [
                    {
                        "index": 1,
                        "formula": "x=1",
                        "dependencies": [2],
                        "final": True,
                    }
                ],
                "threshold": THRESHOLD,
            },
            "02-3006",
            THRESHOLD,
        )

    with pytest.raises(ValueError, match="invalid prepared threshold"):
        static._answer({"formulas": [], "threshold": 0.5}, "02-3006", THRESHOLD)
