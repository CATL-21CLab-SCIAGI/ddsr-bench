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

from ddsr_bench.benchmarks.physics.data.loader import load_split
from ddsr_bench.benchmarks.physics.data.schemas import Problem
from ddsr_bench.benchmarks.physics.evaluation.evaluator import evaluate, unanswered
from ddsr_bench.benchmarks.physics.evaluation.prepare import (
    decode_instruction,
    decode_reference,
)
from ddsr_bench.benchmarks.physics.generation.runner import generate
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
    judge_retries: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate and judge one PHYSICS problem attempt locally."""
    if problem.answer is None:
        raise ValueError(f"PHYSICS problem {problem.spec.id!r} has no reference")
    if resume:
        saved = completed(directory, f"physics/{problem.spec.id}", attempt)
        if saved is not None:
            return saved

    started_at = datetime.now(UTC).isoformat()
    directory.mkdir(parents=True)
    for name in ("agent", "artifacts", "validation"):
        (directory / name).mkdir()
    try:
        response, record = await generate(
            problem.spec,
            client,
            sampling,
            directory / "agent" / "stream.jsonl",
        )
        write_json(directory / "agent" / "response.json", record)
        (directory / "artifacts" / "answer.txt").write_text(response, encoding="utf-8")
    except Exception as error:  # noqa: BLE001 - preserve one failed attempt
        judgment = unanswered(len(problem.answer.final_answers), "no final answer")
        result = {
            "reward": 0.0,
            "mode": "physics",
            "status": "error",
            "error": type(error).__name__,
            "judged": True,
            "skipped": "no final answer",
            "judgment": judgment,
            "split": problem.spec.split,
            "domain": problem.spec.domain,
            "answer_types": list(problem.answer.types),
            "answer_correct": [False] * len(problem.answer.final_answers),
            "confidence": judgment["confidence"],
        }
    else:
        try:
            result = await evaluate(
                response,
                problem.spec,
                problem.answer,
                judge,
                max_tokens=judge_max_tokens,
                retries=judge_retries,
            )
        except Exception as error:  # noqa: BLE001 - preserve one failed attempt
            result = {
                "reward": 0.0,
                "mode": "physics",
                "status": "error",
                "error": type(error).__name__,
                "judged": False,
                "split": problem.spec.split,
                "domain": problem.spec.domain,
                "answer_types": list(problem.answer.types),
                "answer_correct": [False] * len(problem.answer.final_answers),
                "confidence": 100,
            }
    result["judge_model"] = judge_model
    write_json(directory / "validation" / "result.json", result)
    trial = {
        "task_name": f"physics/{problem.spec.id}",
        "trial_name": directory.name,
        "attempt": attempt,
        "started_at": started_at,
        "agent_info": {
            "name": "physics",
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


def _prepared(config: JobConfig, split: str) -> list[Problem]:
    problems = []
    for dataset in config.datasets:
        if dataset.path is None:
            continue
        for instruction in sorted(dataset.path.glob("*/instruction.md")):
            spec = decode_instruction(instruction.read_text(encoding="utf-8"))
            answer = decode_reference(
                (instruction.parent / "tests" / "reference.json").read_text(
                    encoding="utf-8"
                )
            )
            if spec.split == split:
                problems.append(Problem(spec, answer))
    if not problems:
        raise ValueError("no prepared PHYSICS tasks found")
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
    """Run a static PHYSICS job with host-side generation and judging."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("benchmark") != "physics":
        raise ValueError("job benchmark must be 'physics'")
    settings = benchmark_config("physics")
    config = JobConfig.model_validate(raw)
    if len(config.agents) != 1 or config.agents[0].name != "physics":
        raise ValueError("PHYSICS jobs require the static physics agent")
    agent = config.agents[0]
    if not agent.model_name:
        raise ValueError("PHYSICS jobs require a model")

    source = "huggingface"
    try:
        problems = load_split(
            str(settings.get("split", "test")), str(settings["revision"])
        )
    except (ImportError, OSError, ValueError) as source_error:
        try:
            problems = _prepared(config, str(settings.get("split", "test")))
        except (OSError, TypeError, ValueError):
            raise source_error
        source = "prepared"
    if task_name is not None:
        problems = [problem for problem in problems if problem.spec.id == task_name]
        if not problems:
            raise ValueError(f"task {task_name!r} was not found")

    kwargs = agent.kwargs
    include_figures = kwargs.get("include_figures", True)
    if type(include_figures) is not bool:
        raise TypeError("include_figures must be true or false")
    if not include_figures:
        problems = [problem for problem in problems if not problem.spec.figures]
        if not problems:
            raise ValueError("no text-only PHYSICS problems selected")

    output = config.jobs_dir.parent / "static" / config.job_name
    prepare_job(
        output, {**raw, "benchmark_settings": settings}, problems, resume=resume
    )
    client_name = kwargs.get("client_name", "vllm")
    sampling = _sampling(agent.model_name, kwargs.get("sampling", {}))
    judge_config = kwargs.get("judge")
    if not isinstance(judge_config, Mapping):
        raise TypeError("PHYSICS jobs require judge settings")
    judge_name = judge_config.get("client_name", "vllm")
    judge_model = judge_config.get("model")
    if client_name not in CLIENTS or judge_name not in CLIENTS:
        raise ValueError("unknown generation or judge client")
    if not isinstance(judge_model, str) or not judge_model:
        raise ValueError("PHYSICS jobs require a judge model")
    judge_overrides = judge_config.get("sampling", {})
    if not isinstance(judge_overrides, Mapping):
        raise TypeError("judge sampling settings must be a mapping")
    judge_parameters = {
        "max_tokens": int(settings["judge_max_tokens"]),
        "temperature": None,
        "top_p": None,
    }
    judge_parameters.update(judge_overrides)
    judge_sampling = Sampling(judge_model, **judge_parameters)

    generation_connection = (
        nullcontext(client)
        if client is not None
        else CLIENTS[client_name](
            kwargs.get("base_url", "http://127.0.0.1:8000/v1"),
            sampling,
            timeout=float(settings["request_timeout_sec"]),
            stream=bool(kwargs.get("stream", False)),
            api_key=read_api_key(kwargs.get("api_key_env")),
        )
    )
    judge_connection = (
        nullcontext(judge)
        if judge is not None
        else CLIENTS[judge_name](
            judge_config.get("base_url", "http://127.0.0.1:8001/v1"),
            judge_sampling,
            timeout=float(settings["request_timeout_sec"]),
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
                            judge_retries=int(settings["judge_retries"]),
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
        "judged": sum(
            result.get("judged", result["status"] in ("passed", "different"))
            for result in results
        ),
        "passed": sum(result["reward"] == 1 for result in results),
        "errors": sum(result["status"] == "error" for result in results),
    }
