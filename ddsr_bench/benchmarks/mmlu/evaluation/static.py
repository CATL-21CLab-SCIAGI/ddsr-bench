from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from harbor.models.job.config import JobConfig

from ddsr_bench.benchmarks.mmlu.data.loader import load_directory
from ddsr_bench.benchmarks.mmlu.data.schemas import Problem
from ddsr_bench.benchmarks.mmlu.evaluation.evaluator import evaluate
from ddsr_bench.benchmarks.mmlu.evaluation.prepare import (
    decode_instruction,
    decode_reference,
)
from ddsr_bench.benchmarks.mmlu.generation.runner import generate
from ddsr_bench.benchmarks.utils import benchmark_config, write_json
from ddsr_bench.commands.resume import completed, prepare_job
from ddsr_bench.generation.client import CLIENTS, ChatClient, Sampling, read_api_key


def _failure(problem: Problem, error: str) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "mode": "mmlu-generative",
        "status": "error",
        "error": error,
        "subject": problem.spec.subject,
        "category": problem.spec.category,
        "selected": None,
    }


async def run_trial(
    problem: Problem,
    attempt: int,
    directory: Path,
    client: ChatClient,
    sampling: Sampling,
    *,
    client_name: str,
    resume: bool = False,
) -> dict[str, Any]:
    if problem.answer is None:
        raise ValueError(f"MMLU problem {problem.spec.id!r} has no answer")
    if resume:
        saved = completed(directory, f"mmlu/{problem.spec.id}", attempt)
        if saved is not None:
            return saved
    started_at = datetime.now(UTC).isoformat()
    directory.mkdir(parents=True)
    for name in ("agent", "artifacts", "validation"):
        (directory / name).mkdir()
    try:
        response, record = await generate(
            problem.spec, client, sampling, directory / "agent" / "stream.jsonl"
        )
        write_json(directory / "agent" / "response.json", record)
        (directory / "artifacts" / "answer.txt").write_text(response, encoding="utf-8")
        result = evaluate(response, problem.spec, problem.answer)
    except Exception as error:  # noqa: BLE001 - preserve one failed attempt
        result = _failure(problem, type(error).__name__)
    write_json(directory / "validation" / "result.json", result)
    trial = {
        "task_name": f"mmlu/{problem.spec.id}",
        "trial_name": directory.name,
        "attempt": attempt,
        "started_at": started_at,
        "agent_info": {
            "name": "mmlu",
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
    write_json(directory / "result.json", trial)
    return result


def _prepared(config: JobConfig) -> list[Problem]:
    problems = []
    for dataset in config.datasets:
        if dataset.path is None:
            continue
        for instruction in sorted(dataset.path.glob("*/instruction.md")):
            problems.append(
                Problem(
                    decode_instruction(instruction.read_text(encoding="utf-8")),
                    decode_reference(
                        (instruction.parent / "tests" / "reference.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                )
            )
    if not problems:
        raise ValueError("no prepared MMLU tasks found")
    return problems


def _filter(problems: list[Problem], kwargs: Mapping[str, Any]) -> list[Problem]:
    for key, field in (("subjects", "subject"), ("categories", "category")):
        values = kwargs.get(key)
        if values is None:
            continue
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            raise TypeError(f"{key} must be a list of strings")
        problems = [
            problem for problem in problems if getattr(problem.spec, field) in values
        ]
    if not problems:
        raise ValueError("MMLU filters selected no problems")
    return problems


async def run_job(
    path: Path,
    client: ChatClient | None = None,
    *,
    task_name: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one provider-portable, generated-letter MMLU job."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("benchmark") != "mmlu":
        raise ValueError("job benchmark must be 'mmlu'")
    settings = benchmark_config("mmlu")
    config = JobConfig.model_validate(raw)
    if len(config.agents) != 1 or config.agents[0].name != "mmlu":
        raise ValueError("MMLU jobs require its static agent")
    agent = config.agents[0]
    if not agent.model_name:
        raise ValueError("MMLU jobs require a model")
    kwargs = agent.kwargs
    source_path = Path(str(kwargs.get("source_path", "datasets/mmlu/official")))
    try:
        problems = load_directory(source_path)
        source = "dataset"
    except Exception as source_error:  # noqa: BLE001 - use prepared tasks offline
        try:
            problems = _prepared(config)
        except (OSError, TypeError, ValueError):
            raise source_error
        source = "prepared"
    if task_name is not None:
        problems = [problem for problem in problems if problem.spec.id == task_name]
        if not problems:
            raise ValueError(f"task {task_name!r} was not found")
    problems = _filter(problems, kwargs)
    output = config.jobs_dir.parent / "static" / config.job_name
    prepare_job(
        output, {**raw, "benchmark_settings": settings}, problems, resume=resume
    )
    client_name = kwargs.get("client_name", "vllm")
    if client_name not in CLIENTS:
        raise ValueError(f"unknown generation client {client_name!r}")
    sampling_settings = kwargs.get("sampling", {})
    if not isinstance(sampling_settings, Mapping):
        raise TypeError("sampling settings must be a mapping")
    sampling = Sampling(agent.model_name, **dict(sampling_settings))
    connection = (
        nullcontext(client)
        if client is not None
        else CLIENTS[client_name](
            kwargs.get("base_url", "http://127.0.0.1:8000/v1"),
            sampling,
            timeout=float(settings["generation_timeout_sec"]),
            retries=int(settings["transport_retries"]),
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
                            output
                            / f"{quote(problem.spec.id, safe='')}__attempt-{attempt}",
                            active,
                            sampling,
                            client_name=client_name,
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
        "passed": sum(result["status"] == "passed" for result in results),
        "errors": sum(result["status"] == "error" for result in results),
    }
