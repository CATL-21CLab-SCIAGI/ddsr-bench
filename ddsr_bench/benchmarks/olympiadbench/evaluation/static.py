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

from ddsr_bench.benchmarks.olympiadbench.data.loader import load_split
from ddsr_bench.benchmarks.olympiadbench.data.schemas import Problem
from ddsr_bench.benchmarks.olympiadbench.evaluation.evaluator import evaluate, verify
from ddsr_bench.benchmarks.olympiadbench.evaluation.prepare import (
    decode_instruction,
    decode_reference,
)
from ddsr_bench.benchmarks.olympiadbench.generation.runner import generate
from ddsr_bench.benchmarks.utils import benchmark_config, write_json
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
    score_timeout_sec: float = 120,
    resume: bool = False,
) -> dict[str, Any]:
    if problem.answer is None:
        raise ValueError(f"OlympiadBench problem {problem.spec.id!r} has no answer")
    if resume:
        saved = completed(directory, f"olympiadbench/{problem.spec.id}", attempt)
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
        result = await asyncio.to_thread(
            verify,
            response,
            problem.spec,
            problem.answer,
            score_timeout_sec,
        )
    except Exception as error:  # noqa: BLE001 - preserve a failed attempt
        result = evaluate("", problem.spec, problem.answer)
        result["error"] = type(error).__name__
    write_json(directory / "validation" / "result.json", result)
    trial = {
        "task_name": f"olympiadbench/{problem.spec.id}",
        "trial_name": directory.name,
        "attempt": attempt,
        "started_at": started_at,
        "agent_info": {
            "name": "olympiadbench",
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
        raise ValueError("no prepared OlympiadBench tasks found")
    return problems


def _filter(problems: list[Problem], kwargs: Mapping[str, Any]) -> list[Problem]:
    for key, field in (
        ("kinds", "kind"),
        ("modalities", "modality"),
        ("subjects", "subject"),
        ("languages", "language"),
        ("sources", "source"),
    ):
        values = kwargs.get(key)
        if values is None:
            continue
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            raise TypeError(f"{key} must be a list of strings")
        problems = [
            problem for problem in problems if getattr(problem.spec, field) in values
        ]
    include_images = kwargs.get("include_images", True)
    if type(include_images) is not bool:
        raise TypeError("include_images must be true or false")
    if not include_images:
        problems = [problem for problem in problems if not problem.spec.images]
    if not problems:
        raise ValueError("OlympiadBench filters selected no problems")
    return problems


async def run_job(
    path: Path,
    client: ChatClient | None = None,
    *,
    task_name: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("benchmark") != "olympiadbench":
        raise ValueError("job benchmark must be 'olympiadbench'")
    settings = benchmark_config("olympiadbench")
    config = JobConfig.model_validate(raw)
    if len(config.agents) != 1 or config.agents[0].name != "olympiadbench":
        raise ValueError("OlympiadBench jobs require its static agent")
    agent = config.agents[0]
    if not agent.model_name:
        raise ValueError("OlympiadBench jobs require a model")
    source = "huggingface"
    try:
        problems = load_split(str(settings["revision"]))
    except (ImportError, OSError, ValueError) as source_error:
        try:
            problems = _prepared(config)
        except (OSError, TypeError, ValueError):
            raise source_error
        source = "prepared"
    if task_name is not None:
        problems = [problem for problem in problems if problem.spec.id == task_name]
        if not problems:
            raise ValueError(f"task {task_name!r} was not found")
    kwargs = agent.kwargs
    problems = _filter(problems, kwargs)
    output = config.jobs_dir.parent / "static" / config.job_name
    prepare_job(
        output, {**raw, "benchmark_settings": settings}, problems, resume=resume
    )
    client_name = kwargs.get("client_name", "vllm")
    if client_name not in CLIENTS:
        raise ValueError("unknown generation client")
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
                            score_timeout_sec=float(settings["score_timeout_sec"]),
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
        "graded": sum(result["status"] != "ungraded" for result in results),
        "passed": sum(result["status"] == "passed" for result in results),
        "ungraded": sum(result["status"] == "ungraded" for result in results),
        "errors": sum(result["status"] == "error" for result in results),
    }
