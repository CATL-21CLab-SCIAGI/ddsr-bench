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

from ddsr_bench.benchmarks.mmmupro.data.loader import OFFICIAL_SETTINGS, load_directory
from ddsr_bench.benchmarks.mmmupro.data.schemas import Mode, Problem
from ddsr_bench.benchmarks.mmmupro.evaluation.evaluator import evaluate
from ddsr_bench.benchmarks.mmmupro.evaluation.prepare import (
    decode_instruction,
    decode_reference,
)
from ddsr_bench.benchmarks.mmmupro.generation.runner import generate
from ddsr_bench.benchmarks.utils import benchmark_config, write_json
from ddsr_bench.commands.resume import completed, prepare_job
from ddsr_bench.generation.client import CLIENTS, ChatClient, Sampling, read_api_key


def _failure(problem: Problem, error: str) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "mode": "mmmupro",
        "status": "error",
        "error": error,
        "setting": problem.spec.setting,
        "subject": problem.spec.subject,
        "domain": problem.spec.domain,
        "selected": None,
        "method": None,
    }


async def run_trial(
    problem: Problem,
    mode: Mode,
    attempt: int,
    directory: Path,
    client: ChatClient,
    sampling: Sampling,
    *,
    client_name: str,
    resume: bool = False,
) -> dict[str, Any]:
    if problem.answer is None:
        raise ValueError(f"MMMU-Pro problem {problem.spec.id!r} has no answer")
    if resume:
        saved = completed(directory, f"mmmupro/{problem.spec.id}", attempt)
        if saved is not None:
            return saved
    started_at = datetime.now(UTC).isoformat()
    directory.mkdir(parents=True)
    for name in ("agent", "artifacts", "validation"):
        (directory / name).mkdir()
    try:
        response, record = await generate(
            problem.spec,
            mode,
            client,
            sampling,
            directory / "agent" / "stream.jsonl",
        )
        write_json(directory / "agent" / "response.json", record)
        (directory / "artifacts" / "answer.txt").write_text(response, encoding="utf-8")
        result = evaluate(response, problem.spec, problem.answer)
    except Exception as error:  # noqa: BLE001 - preserve one failed attempt
        result = _failure(problem, type(error).__name__)
    write_json(directory / "validation" / "result.json", result)
    trial = {
        "task_name": f"mmmupro/{problem.spec.id}",
        "trial_name": directory.name,
        "attempt": attempt,
        "started_at": started_at,
        "agent_info": {
            "name": "mmmupro",
            "model_info": {"provider": client_name, "name": sampling.model},
        },
        "config": {
            "agent": {
                "kwargs": {
                    "client_name": client_name,
                    "sampling": asdict(sampling),
                    "mode": mode,
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
        raise ValueError("no prepared MMMU-Pro tasks found")
    return problems


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a list of strings")
    return tuple(value)


def _filter(problems: list[Problem], kwargs: Mapping[str, Any]) -> list[Problem]:
    for key, field in (("subjects", "subject"), ("domains", "domain")):
        values = kwargs.get(key)
        if values is None:
            continue
        selected = _strings(values, key)
        problems = [
            problem for problem in problems if getattr(problem.spec, field) in selected
        ]
    if not problems:
        raise ValueError("MMMU-Pro filters selected no problems")
    return problems


async def run_job(
    path: Path,
    client: ChatClient | None = None,
    *,
    task_name: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one static multimodal MMMU-Pro job."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("benchmark") != "mmmupro":
        raise ValueError("job benchmark must be 'mmmupro'")
    settings = benchmark_config("mmmupro")
    config = JobConfig.model_validate(raw)
    if len(config.agents) != 1 or config.agents[0].name != "mmmupro":
        raise ValueError("MMMU-Pro jobs require its static agent")
    agent = config.agents[0]
    if not agent.model_name:
        raise ValueError("MMMU-Pro jobs require a model")
    kwargs = agent.kwargs
    selected_settings = _strings(
        kwargs.get("settings", list(OFFICIAL_SETTINGS)), "settings"
    )
    mode = kwargs.get("mode", settings["default_mode"])
    if mode not in ("cot", "direct"):
        raise ValueError("MMMU-Pro mode must be 'cot' or 'direct'")
    source_path = Path(str(kwargs.get("source_path", "datasets/mmmupro/corrected")))
    try:
        problems = load_directory(source_path, selected_settings)
        source = "dataset"
    except Exception as source_error:  # noqa: BLE001 - use prepared tasks offline
        try:
            problems = _prepared(config)
        except (OSError, TypeError, ValueError):
            raise source_error
        problems = [
            problem for problem in problems if problem.spec.setting in selected_settings
        ]
        if not problems:
            raise ValueError("prepared tasks do not contain selected MMMU-Pro settings")
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
                            mode,
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
