import asyncio
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import pytest
import yaml

from ddsr_bench.commands.resume import completed, prepare_job
from ddsr_bench.commands.solve import run_job
from ddsr_bench.generation.client import ChatResponse


@dataclass
class Problem:
    statement: str


def test_job_settings(tmp_path: Path) -> None:
    output = tmp_path / "job"
    config = {"model": "teacher", "n_concurrent_trials": 1}
    problems = [Problem("question")]
    prepare_job(output, config, problems, resume=False)
    prepare_job(output, dict(config, n_concurrent_trials=4), problems, resume=True)
    with pytest.raises(ValueError, match="unchanged"):
        prepare_job(output, dict(config, model="other"), problems, resume=True)
    with pytest.raises(ValueError, match="unchanged"):
        prepare_job(output, config, [Problem("changed")], resume=True)
    with pytest.raises(FileExistsError):
        prepare_job(output, config, problems, resume=False)


def test_unfinished_trial(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "partial.txt").write_text("saved stream")
    assert completed(trial, "benchmark/1", 0) is None
    assert not trial.exists()
    archived = list((tmp_path / ".interrupted").glob("*/partial.txt"))
    assert len(archived) == 1
    assert archived[0].read_text() == "saved stream"


def test_truncated_result(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "result.json").write_text('{"task_name":')
    assert completed(trial, "cmphysbench/7", 0) is None
    assert not trial.exists()
    assert (
        next((tmp_path / ".interrupted").glob("*/result.json")).read_text()
        == '{"task_name":'
    )


@pytest.mark.parametrize("status", ["passed", "error"])
def test_completed_trial(tmp_path: Path, status: str) -> None:
    result = {"status": status, "reward": 0}
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "task_name": "benchmark/1",
                "attempt": 0,
                "static_result": result,
            }
        )
    )
    assert completed(tmp_path, "benchmark/1", 0) == result
    with pytest.raises(ValueError, match="identity"):
        completed(tmp_path, "benchmark/2", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["cmphysbench", "phybench"])
async def test_job_recovery(tmp_path, monkeypatch, benchmark):
    static = import_module(f"ddsr_bench.benchmarks.{benchmark}.evaluation.static")
    loader = import_module(f"ddsr_bench.benchmarks.{benchmark}.data.loader")
    if benchmark == "phybench":
        fixture = Path(__file__).parents[1] / "fixtures" / "phybench_133.json"
        problem = loader.load_problem(json.loads(fixture.read_text()))
    else:
        problem = loader.load_problem(
            {
                "id": 7,
                "question": "Find the ratio.",
                "symbol": "$x$, $y$",
                "answer_type": "Expression",
                "topic": "Theoretical Foundations",
                "final_answer": ["x^2/y"],
            }
        )
    monkeypatch.setattr(static, "load_split", lambda **_: [problem])
    monkeypatch.setattr(
        static,
        "verify",
        lambda *_: {
            "status": "passed",
            "reward": 1.0,
        },
    )

    class Client:
        calls = 0
        interrupt = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def preflight(self):
            pass

        async def chat(self, *args, **kwargs):
            self.calls += 1
            if self.interrupt and self.calls == 2:
                raise asyncio.CancelledError
            return ChatResponse(
                content="answer",
                reasoning=None,
                model="teacher",
                usage={},
                seed=None,
                latency=0.0,
                raw={},
                finish_reason="stop",
            )

    client = Client()
    monkeypatch.setitem(static.CLIENTS, "vllm", lambda *a, **k: client)

    name = "CMPhysBenchAgent" if benchmark == "cmphysbench" else "PHYBenchAgent"
    config = {
        "benchmark": benchmark,
        "job_name": "resume-test",
        "jobs_dir": str(tmp_path / "harbor"),
        "n_attempts": 2,
        "n_concurrent_trials": 1,
        "agents": [
            {
                "name": f"ddsr_bench.benchmarks.{benchmark}.evaluation.harbor:{name}",
                "model_name": "teacher",
            }
        ],
    }
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(asyncio.CancelledError):
        await run_job(path)
    output = tmp_path / "static" / "resume-test"
    first = output / f"{problem.spec.id}__attempt-0" / "result.json"
    saved = first.read_bytes()
    assert client.calls == 2
    client.interrupt = False
    config["n_concurrent_trials"] = 2
    path.write_text(yaml.safe_dump(config))
    result = await run_job(path, resume=True)
    assert result["passed"] == 2
    assert client.calls == 3  # Only the unfinished trial was regenerated.
    assert first.read_bytes() == saved
    assert len(list((output / ".interrupted").iterdir())) == 1
    await run_job(path, resume=True)
    assert client.calls == 3
    config["agents"][0]["model_name"] = "different-model"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="unchanged"):
        await run_job(path, resume=True)
    assert client.calls == 3
