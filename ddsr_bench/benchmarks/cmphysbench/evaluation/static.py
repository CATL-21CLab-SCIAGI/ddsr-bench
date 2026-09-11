from __future__ import annotations

import asyncio
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path
from typing import Any

import yaml
from harbor.models.job.config import JobConfig

from ddsr_bench.benchmarks.cmphysbench.data.loader import DATASET, REVISION, load_split
from ddsr_bench.benchmarks.cmphysbench.data.schemas import AnswerSpec, Problem
from ddsr_bench.benchmarks.cmphysbench.evaluation.prepare import decode_instruction
from ddsr_bench.benchmarks.cmphysbench.evaluation.verifier import verify
from ddsr_bench.benchmarks.cmphysbench.generation.runner import generate
from ddsr_bench.benchmarks.registry import benchmark_config
from ddsr_bench.generation.client import CLIENTS, ChatClient, Sampling


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


async def run_trial(
    problem: Problem,
    attempt: int,
    directory: Path,
    client: ChatClient,
    sampling: Sampling,
    *,
    client_name: str,
    timeout_sec: float = 120,
) -> dict[str, Any]:
    """Generate and score one CMPhysBench problem attempt locally."""
    if problem.answer is None:
        raise ValueError(f"CMPhysBench problem {problem.spec.id!r} has no reference")

    started_at = datetime.now(UTC).isoformat()
    directory.mkdir(parents=True)
    agent_dir = directory / "agent"
    artifacts = directory / "artifacts"
    validation = directory / "validation"
    for path in (agent_dir, artifacts, validation):
        path.mkdir()

    try:
        response, record = await generate(problem.spec, client, sampling)
        _write(agent_dir / "response.json", record)
        (artifacts / "answer.txt").write_text(response, encoding="utf-8")
        result = await asyncio.to_thread(
            verify,
            response,
            problem.answer.value,
            problem.spec.answer_type,
            timeout_sec,
        )
    except Exception as error:  # noqa: BLE001 - one failed attempt receives zero
        result = {
            "reward": 0.0,
            "mode": "seed",
            "status": "error",
            "error": type(error).__name__,
        }

    result |= {
        "answer_type": problem.spec.answer_type,
        "topic": problem.spec.topic,
    }
    _write(validation / "result.json", result)
    trial = {
        "task_name": f"cmphysbench/{problem.spec.id}",
        "trial_name": directory.name,
        "attempt": attempt,
        "started_at": started_at,
        "agent_info": {
            "name": "cmphysbench",
            "model_info": {"provider": client_name, "name": sampling.model},
        },
        "config": {
            "agent": {
                "kwargs": {
                    "client_name": client_name,
                    "sampling": asdict(sampling),
                }
            }
        },
        "static_result": result,
    }
    _write(directory / "result.json", trial)
    return result


def _api_key(name: str | None) -> str | None:
    if name is None:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"environment variable {name!r} is not set")
    return value


def _prepared(config: JobConfig) -> list[Problem]:
    problems = []
    for dataset in config.datasets:
        if dataset.path is None:
            continue
        for instruction in sorted(dataset.path.glob("*/instruction.md")):
            spec = decode_instruction(instruction.read_text(encoding="utf-8"))
            reference = json.loads(
                (instruction.parent / "tests" / "reference.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                not isinstance(reference, dict)
                or not isinstance(reference.get("answer"), str)
                or reference.get("answer_type") != spec.answer_type
            ):
                raise ValueError(f"invalid prepared reference for {spec.id!r}")
            problems.append(Problem(spec, AnswerSpec(reference["answer"])))
    if not problems:
        raise ValueError("no prepared CMPhysBench tasks found")
    return problems


async def run_job(
    path: Path,
    client: ChatClient | None = None,
    *,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Run a CMPhysBench static job directly from its pinned dataset."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("benchmark") != "cmphysbench":
        raise ValueError("job benchmark must be 'cmphysbench'")
    benchmark = benchmark_config("cmphysbench")
    if benchmark.get("dataset") != DATASET or benchmark.get("revision") != REVISION:
        raise ValueError("CMPhysBench configuration must use the pinned dataset")
    if benchmark.get("gradable_only") is not True:
        raise ValueError("CMPhysBench evaluation requires gradable_only: true")

    config = JobConfig.model_validate(raw)
    if len(config.agents) != 1:
        raise ValueError("CMPhysBench jobs require exactly one agent")
    agent = config.agents[0]
    expected_agent = (
        "ddsr_bench.benchmarks.cmphysbench.evaluation.harbor:CMPhysBenchAgent"
    )
    if agent.name != expected_agent or not agent.model_name:
        raise ValueError("CMPhysBench job requires its named agent and model")

    source = "huggingface"
    try:
        problems = load_split(
            split=str(benchmark["split"]),
            revision=str(benchmark["revision"]),
            gradable_only=True,
        )
    except OSError:
        problems = _prepared(config)
        source = "prepared"
    if task_name is not None:
        problems = [problem for problem in problems if problem.spec.id == task_name]
        if not problems:
            raise ValueError(f"task {task_name!r} was not found")
    output = config.jobs_dir.parent / "static" / config.job_name
    output.mkdir(parents=True)
    kwargs = agent.kwargs
    client_name = kwargs.get("client_name", "vllm")
    if client_name not in CLIENTS:
        raise ValueError(f"unknown client {client_name!r}")
    sampling = Sampling(agent.model_name, **dict(kwargs.get("sampling", {})))
    connection = (
        nullcontext(client)
        if client is not None
        else CLIENTS[client_name](
            kwargs.get("base_url", "http://127.0.0.1:8000/v1"),
            sampling,
            stream=bool(kwargs.get("stream", False)),
            api_key=_api_key(kwargs.get("api_key_env")),
        )
    )
    work = [
        (problem, attempt)
        for problem in problems
        for attempt in range(config.n_attempts)
    ]
    results = []
    async with connection as active:
        await active.preflight()
        for group in batched(work, config.n_concurrent_trials):
            results.extend(
                await asyncio.gather(
                    *(
                        run_trial(
                            problem,
                            attempt,
                            output / f"{problem.spec.id}__attempt-{attempt}",
                            active,
                            sampling,
                            client_name=client_name,
                            timeout_sec=config.verifier.override_timeout_sec or 120,
                        )
                        for problem, attempt in group
                    )
                )
            )
    return {
        "output": str(output),
        "source": source,
        "trials": len(results),
        "scored": sum(
            result["status"] in ("passed", "different") for result in results
        ),
        "passed": sum(result["reward"] == 1 for result in results),
        "errors": sum(result["status"] in ("error", "timeout") for result in results),
    }
