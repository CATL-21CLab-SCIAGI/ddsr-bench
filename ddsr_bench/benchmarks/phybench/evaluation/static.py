from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path
from typing import Any

import yaml
from harbor.models.job.config import JobConfig

from ddsr_bench.benchmarks.phybench.data.loader import (
    DATA_FILE,
    DATASET,
    REVISION,
    load_split,
)
from ddsr_bench.benchmarks.phybench.data.schemas import AnswerSpec, Problem
from ddsr_bench.benchmarks.phybench.evaluation.prepare import decode_instruction
from ddsr_bench.benchmarks.phybench.evaluation.verifier import verify
from ddsr_bench.benchmarks.phybench.generation.runner import generate
from ddsr_bench.benchmarks.registry import benchmark_config
from ddsr_bench.benchmarks.utils import write_json as _write
from ddsr_bench.commands.resume import completed, prepare_job
from ddsr_bench.generation.client import CLIENTS, ChatClient, Sampling, read_api_key


async def run_trial(
    problem: Problem,
    attempt: int,
    directory: Path,
    client: ChatClient,
    sampling: Sampling,
    *,
    client_name: str,
    timeout_sec: float = 120,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate and score one PHYBench problem attempt locally."""
    if problem.answer is None:
        raise ValueError(f"PHYBench problem {problem.spec.id!r} has no reference")

    if resume:
        saved = completed(directory, f"phybench/{problem.spec.id}", attempt)
        if saved is not None:
            return saved
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
            verify, response, problem.answer.value, timeout_sec
        )
    except Exception as error:  # noqa: BLE001 - one failed attempt receives zero
        result = {
            "reward": 0.0,
            "mode": "eed",
            "status": "error",
            "error": type(error).__name__,
        }

    result["tag"] = problem.spec.tag
    _write(validation / "result.json", result)
    trial = {
        "task_name": f"phybench/{problem.spec.id}",
        "trial_name": directory.name,
        "attempt": attempt,
        "started_at": started_at,
        "agent_info": {
            "name": "phybench",
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
            if not isinstance(reference, dict) or not isinstance(
                reference.get("answer"), str
            ):
                raise TypeError(f"invalid prepared reference for {spec.id!r}")
            problems.append(Problem(spec, AnswerSpec(reference["answer"], solution="")))
    if not problems:
        raise ValueError("no prepared PHYBench tasks found")
    return problems


async def run_job(
    path: Path,
    client: ChatClient | None = None,
    *,
    task_name: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run a PHYBench static job from pinned data or prepared tasks."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("benchmark") != "phybench":
        raise ValueError("job benchmark must be 'phybench'")
    settings = benchmark_config("phybench")
    configured = (
        settings.get("dataset"),
        settings.get("revision"),
        settings.get("data_file"),
        settings.get("split"),
        settings.get("gradable_only"),
    )
    if configured != (DATASET, REVISION, DATA_FILE, "train", True):
        raise ValueError("PHYBench configuration must use its gradable pinned data")

    config = JobConfig.model_validate(raw)
    if len(config.agents) != 1:
        raise ValueError("PHYBench jobs require exactly one agent")
    agent = config.agents[0]
    expected = "ddsr_bench.benchmarks.phybench.evaluation.harbor:PHYBenchAgent"
    if agent.name != expected or not agent.model_name:
        raise ValueError("PHYBench job requires its named agent and model")

    source = "huggingface"
    try:
        problems = load_split(
            split=str(settings["split"]),
            revision=str(settings["revision"]),
            gradable_only=True,
        )
    except (ImportError, OSError, ValueError) as source_error:
        try:
            problems = _prepared(config)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            raise source_error
        source = "prepared"
    if task_name is not None:
        problems = [problem for problem in problems if problem.spec.id == task_name]
        if not problems:
            raise ValueError(f"task {task_name!r} was not found")

    output = config.jobs_dir.parent / "static" / config.job_name
    prepare_job(
        output, {**raw, "benchmark_settings": settings}, problems, resume=resume
    )
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
            api_key=read_api_key(kwargs.get("api_key_env")),
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
                            resume=resume,
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
