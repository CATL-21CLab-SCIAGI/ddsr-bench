import asyncio
import json
from pathlib import Path

import pytest
from harbor.models.job.config import AgentConfig

from ddsr_bench.benchmarks.collect import collect_trials
from ddsr_bench.benchmarks.critpt.data.schemas import ProblemSpec
from ddsr_bench.benchmarks.critpt.evaluation.prepare import encode_instruction
from ddsr_bench.benchmarks.critpt.evaluation.static import (
    run_job,
    run_trial,
    trial_seed,
)
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


class OptionClient(FakeClient):
    async def chat(self, messages: ChatMessages, **kwargs) -> ChatResponse:
        return await super().chat(messages)


def test_trial_seeds_are_stable_and_distinct():
    work = [(f"Challenge_{i}_main", a) for i in range(1, 71) for a in range(5)]
    seeds = {(p, a): trial_seed(42, p, a) for p, a in work}
    assert len(set(seeds.values())) == 350
    assert seeds == {(p, a): trial_seed(42, p, a) for p, a in reversed(work)}
    assert seeds[("Challenge_1_main", 0)] != trial_seed(43, "Challenge_1_main", 0)


@pytest.mark.asyncio
async def test_seeded_trial_still_formats_empty_first_stage(tmp_path):
    class Client:
        def __init__(self):
            self.seeds = []

        async def chat(self, messages, **kwargs):
            self.seeds.append(kwargs["seed"])
            first = len(self.seeds) == 1
            if not first:
                assert messages[-2] == {"role": "assistant", "content": ""}
                assert kwargs["max_tokens"] == 65536
            return ChatResponse(
                "" if first else "def answer():\n    return 42\n",
                "thought",
                "local",
                None,
                kwargs["seed"],
                0.1,
                {},
                "length" if first else "stop",
            )

    problem = ProblemSpec(
        "p1",
        "main",
        None,
        "Return 42.",
        "def answer():\n    return ...\n",
        "critpt",
        Path("source.json"),
    )
    agent = AgentConfig(
        name="ddsr_bench.benchmarks.critpt.evaluation.harbor:CritPtAgent",
        model_name="local",
        kwargs={
            "style": "two-step",
            "seed_base": 42,
            "require_complete_stages": False,
            "formatting_max_tokens": 65536,
        },
    )
    seeds = []
    for attempt, name in [(0, "original"), (0, "replay"), (1, "next")]:
        client = Client()
        output = tmp_path / name
        assert await run_trial(problem, attempt, output, client, agent) == "validated"
        expected = trial_seed(42, "p1", attempt)
        assert client.seeds == [expected, expected]
        record = json.loads((output / "agent/response.json").read_text())
        assert record["seed"] == expected
        assert record["responses"][0]["finish_reason"] == "length"
        assert len(record["responses"]) == 2
        seeds.append(expected)
    assert seeds[0] == seeds[1] != seeds[2]
    with pytest.raises(TypeError, match="seed_base requires"):
        await run_trial(problem, 0, tmp_path / "unsupported", FakeClient(), agent)


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
        name="ddsr_bench.benchmarks.critpt.evaluation.harbor:CritPtAgent",
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
        name="ddsr_bench.benchmarks.critpt.evaluation.harbor:CritPtAgent",
        model_name="local",
    )
    output = tmp_path / "p1__attempt-0"

    status = await run_trial(problem, 0, output, InvalidClient(), agent)

    result = json.loads((output / "validation" / "result.json").read_text())
    assert status == "error"
    assert result["reward"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["vllm", "openai"])
@pytest.mark.parametrize("seed_base", [None, 42])
async def test_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider, seed_base
) -> None:
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
    config.write_text(
        """job_name: batch
jobs_dir: outputs/harbor
n_attempts: 2
n_concurrent_trials: 2
agents:
  - name: ddsr_bench.benchmarks.critpt.evaluation.harbor:CritPtAgent
    model_name: local
    kwargs: {client_name: PROVIDER, style: one-step, seed_base: SEED}
datasets:
  - path: tasks
""".replace("PROVIDER", provider).replace(
            "SEED", "null" if seed_base is None else str(seed_base)
        )
    )
    monkeypatch.chdir(tmp_path)
    client = FakeClient() if seed_base is None else OptionClient()

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
    for attempt in range(2):
        record = json.loads(
            (output / f"p1__attempt-{attempt}/agent/response.json").read_text()
        )
        expected_seed = (
            None if seed_base is None else trial_seed(seed_base, "p1", attempt)
        )
        assert record["seed"] == expected_seed

    summary = collect_trials(output)
    assert summary["complete_batches"] == 2
    assert summary["batches"][0]["validated"] == 1
    assert summary["batches"][0]["trials"][0]["reward"] is None

    # Finished answers survive a concurrency change without another model call.
    config.write_text(
        config.read_text().replace("n_concurrent_trials: 2", "n_concurrent_trials: 4")
    )
    replay = await run_job(config, InvalidClient(), resume=True)
    assert replay == result
    config.write_text(config.read_text().replace("n_attempts: 2", "n_attempts: 3"))
    with pytest.raises(ValueError, match="only permits changing n_concurrent_trials"):
        await run_job(config, FakeClient(), resume=True)


@pytest.mark.asyncio
async def test_job_refills_free_slot_while_another_trial_is_still_running(
    tmp_path, monkeypatch
):
    for i in range(3):
        task = tmp_path / "tasks" / f"p{i}"
        task.mkdir(parents=True)
        problem = ProblemSpec(
            f"p{i}",
            "main",
            None,
            "Return 42.",
            "def answer():\n    return ...\n",
            "critpt",
            Path("source.json"),
        )
        (task / "instruction.md").write_text(encode_instruction(problem))
    config = tmp_path / "job.yaml"
    config.write_text("""job_name: refill
jobs_dir: outputs/harbor
n_attempts: 1
n_concurrent_trials: 2
agents:
  - name: ddsr_bench.benchmarks.critpt.evaluation.harbor:CritPtAgent
    model_name: local
datasets:
  - path: tasks
""")
    monkeypatch.chdir(tmp_path)
    third_started = asyncio.Event()
    running = 0
    peak = 0

    async def trial(problem, attempt, directory, client, agent):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        try:
            if problem.id == "p0":
                await asyncio.wait_for(third_started.wait(), timeout=2)
            elif problem.id == "p2":
                third_started.set()
            return "error" if problem.id == "p1" else "validated"
        finally:
            running -= 1

    monkeypatch.setattr(
        "ddsr_bench.benchmarks.critpt.evaluation.static.run_trial", trial
    )
    result = await run_job(config, FakeClient())
    assert peak == 2
    assert result["validated"] == 2
    assert result["errors"] == 1
