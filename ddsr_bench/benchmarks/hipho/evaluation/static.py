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

from ddsr_bench.benchmarks.hipho.data.loader import PUBLISHED_EXAMS, load_split
from ddsr_bench.benchmarks.hipho.data.schemas import Problem
from ddsr_bench.benchmarks.hipho.evaluation.evaluator import evaluate, unanswered
from ddsr_bench.benchmarks.hipho.evaluation.prepare import (
    decode_instruction,
    decode_reference,
)
from ddsr_bench.benchmarks.hipho.generation.runner import generate
from ddsr_bench.benchmarks.utils import benchmark_config, write_json
from ddsr_bench.commands.resume import completed, prepare_job
from ddsr_bench.generation.client import CLIENTS, ChatClient, Sampling, read_api_key


async def run_trial(
    problem: Problem,
    attempt: int,
    directory: Path,
    client: ChatClient,
    judge: ChatClient,
    sampling: Sampling,
    *,
    client_name: str,
    judge_name: str,
    judge_model: str,
    judge_max_tokens: int,
    resume: bool = False,
) -> dict[str, Any]:
    if problem.answer is None:
        raise ValueError(f"HiPhO problem {problem.spec.id!r} has no answer")
    if resume:
        saved = completed(directory, f"hipho/{problem.spec.id}", attempt)
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
        result = await evaluate(
            response,
            problem.spec,
            problem.answer,
            judge,
            max_tokens=judge_max_tokens,
        )
    except Exception as error:  # noqa: BLE001 - preserve one failed attempt
        result = unanswered(problem.spec, type(error).__name__)
        result["max_points"] = sum(problem.answer.points)
    result["judge_model"] = judge_model
    write_json(directory / "validation" / "result.json", result)
    trial = {
        "task_name": f"hipho/{problem.spec.id}",
        "trial_name": directory.name,
        "attempt": attempt,
        "started_at": started_at,
        "agent_info": {
            "name": "hipho",
            "model_info": {"provider": client_name, "name": sampling.model},
        },
        "config": {
            "agent": {
                "kwargs": {
                    "client_name": client_name,
                    "sampling": asdict(sampling),
                    "judge": {"client_name": judge_name, "model": judge_model},
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
        raise ValueError("no prepared HiPhO tasks found")
    return problems


def _filter(problems: list[Problem], kwargs: Mapping[str, Any]) -> list[Problem]:
    cohort = kwargs.get("cohort", "current")
    if cohort not in ("current", "published"):
        raise ValueError("cohort must be 'current' or 'published'")
    if cohort == "published":
        problems = [
            problem for problem in problems if problem.spec.exam in PUBLISHED_EXAMS
        ]
    for key, field in (
        ("exams", "exam"),
        ("fields", "field"),
        ("modalities", "modality"),
        ("languages", "language"),
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
        raise ValueError("HiPhO filters selected no problems")
    return problems


def _sampling(model: str, value: Any) -> Sampling:
    if not isinstance(value, Mapping):
        raise TypeError("sampling settings must be a mapping")
    return Sampling(model, **dict(value))


async def run_job(
    path: Path,
    client: ChatClient | None = None,
    judge: ChatClient | None = None,
    *,
    task_name: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run a static HiPhO job with the maintained official scorer."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("benchmark") != "hipho":
        raise ValueError("job benchmark must be 'hipho'")
    settings = benchmark_config("hipho")
    config = JobConfig.model_validate(raw)
    if len(config.agents) != 1 or config.agents[0].name != "hipho":
        raise ValueError("HiPhO jobs require its static agent")
    agent = config.agents[0]
    if not agent.model_name:
        raise ValueError("HiPhO jobs require a model")
    source = "huggingface"
    try:
        problems = load_split(str(settings["revision"]))
    except Exception as source_error:  # noqa: BLE001 - try the local mirror offline
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
    sampling = _sampling(agent.model_name, kwargs.get("sampling", {}))
    judge_config = kwargs.get("judge")
    if not isinstance(judge_config, Mapping):
        raise TypeError("HiPhO jobs require judge settings")
    judge_name = judge_config.get("client_name", "openai")
    judge_model = judge_config.get("model", settings["judge_model"])
    if client_name not in CLIENTS or judge_name not in CLIENTS:
        raise ValueError("unknown generation or judge client")
    if not isinstance(judge_model, str) or not judge_model:
        raise ValueError("HiPhO jobs require a judge model")
    judge_overrides = judge_config.get("sampling", {})
    if not isinstance(judge_overrides, Mapping):
        raise TypeError("judge sampling settings must be a mapping")
    judge_parameters = {
        "max_tokens": int(settings["judge_max_tokens"]),
        "temperature": float(settings["judge_temperature"]),
        "top_p": 1.0,
    }
    judge_parameters.update(judge_overrides)
    judge_sampling = Sampling(judge_model, **judge_parameters)
    generation_connection = (
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
    judge_connection = (
        nullcontext(judge)
        if judge is not None
        else CLIENTS[judge_name](
            judge_config.get(
                "base_url", "https://generativelanguage.googleapis.com/v1beta/openai"
            ),
            judge_sampling,
            timeout=float(settings["judge_timeout_sec"]),
            retries=int(settings["transport_retries"]),
            api_key=read_api_key(judge_config.get("api_key_env")),
        )
    )
    work = [
        (problem, attempt)
        for problem in problems
        for attempt in range(config.n_attempts)
    ]
    results = []
    async with generation_connection as active, judge_connection as active_judge:
        await active.preflight()
        await active_judge.preflight()
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
                            active_judge,
                            sampling,
                            client_name=client_name,
                            judge_name=judge_name,
                            judge_model=judge_model,
                            judge_max_tokens=judge_sampling.max_tokens,
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
        "judged": sum(result.get("judged", False) for result in results),
        "passed": sum(result["status"] == "passed" for result in results),
        "errors": sum(result["status"] == "error" for result in results),
    }
