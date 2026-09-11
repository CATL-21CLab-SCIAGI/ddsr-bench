"""Run prepared CritPt tasks without executing generated code."""

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
from harbor.models.job.config import AgentConfig, JobConfig

from ddsr_bench.benchmarks.critpt.data.schemas import ProblemSpec
from ddsr_bench.benchmarks.critpt.generation.prompts import PromptStyle
from ddsr_bench.benchmarks.critpt.generation.runner import generate
from ddsr_bench.generation.client import CLIENTS, ChatClient, Sampling
from ddsr_bench.grading.validation import extract_answer

from .prepare import decode_instruction


async def run_trial(
    problem: ProblemSpec,
    attempt: int,
    directory: Path,
    client: ChatClient,
    agent: AgentConfig,
) -> str:
    """Generate and statically validate one problem attempt."""
    sampling = Sampling(agent.model_name, **dict(agent.kwargs.get("sampling", {})))
    style: PromptStyle = agent.kwargs.get("style", "one-step")
    client_name = agent.kwargs.get("client_name", "vllm")
    agent_dir = directory / "agent"
    artifacts = directory / "artifacts"
    validation = directory / "validation"
    for path in (agent_dir, artifacts, validation):
        path.mkdir(parents=True)

    try:
        response, record = await generate(client, problem, style, sampling)
        (agent_dir / "response.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        code = extract_answer(response, problem.code_template)
        (artifacts / "answer.py").write_text(code, encoding="utf-8")
        result: dict[str, Any] = {
            "reward": None,
            "mode": "static",
            "status": "validated",
        }
    except Exception as error:  # noqa: BLE001 - isolate failed model attempts
        result = {
            "reward": None,
            "mode": "static",
            "status": "error",
            "error": type(error).__name__,
            "message": str(error),
        }

    (validation / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    trial = {
        "task_name": f"critpt/{problem.id}",
        "trial_name": directory.name,
        "attempt": attempt,
        "started_at": datetime.now(UTC).isoformat(),
        "agent_info": {
            "name": "critpt",
            "model_info": {"provider": client_name, "name": sampling.model},
        },
        "config": {
            "agent": {
                "kwargs": {
                    "client_name": client_name,
                    "style": style,
                    "sampling": asdict(sampling),
                }
            }
        },
        "static_result": result,
    }
    (directory / "result.json").write_text(
        json.dumps(trial, indent=2), encoding="utf-8"
    )
    return result["status"]


async def run_job(
    path: Path,
    client: ChatClient | None = None,
    *,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Run every task in one Harbor job config without code execution."""
    config = JobConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if len(config.agents) != 1:
        raise ValueError("static runs require exactly one agent")
    agent = config.agents[0]
    if (
        agent.name != "ddsr_bench.benchmarks.critpt.evaluation.harbor:CritPtAgent"
        or not agent.model_name
    ):
        raise ValueError("static runs require CritPtAgent and a model name")

    problems = []
    for dataset in config.datasets:
        if dataset.path is None:
            raise ValueError("static runs require local dataset paths")
        problems.extend(
            decode_instruction(file.read_text(encoding="utf-8"))
            for file in sorted(dataset.path.glob("*/instruction.md"))
        )
    ids = [problem.id for problem in problems]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("static tasks must contain unique problem IDs")
    if task_name is not None:
        problems = [problem for problem in problems if problem.id == task_name]
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

    statuses = []
    work = [
        (problem, attempt)
        for problem in problems
        for attempt in range(config.n_attempts)
    ]
    async with connection as active:
        await active.preflight()
        for group in batched(work, config.n_concurrent_trials):
            statuses.extend(
                await asyncio.gather(
                    *(
                        run_trial(
                            problem,
                            attempt,
                            output / f"{problem.id}__attempt-{attempt}",
                            active,
                            agent,
                        )
                        for problem, attempt in group
                    )
                )
            )
    return {
        "output": str(output),
        "trials": len(statuses),
        "validated": statuses.count("validated"),
        "errors": statuses.count("error"),
    }


def _api_key(name: str | None) -> str | None:
    if not name:
        return None
    import os

    value = os.environ.get(name)
    if not value:
        raise ValueError(f"environment variable {name!r} is not set")
    return value
