"""Run prepared CritPt tasks without executing generated code."""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import UTC, datetime
from hashlib import sha256
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
from .resume import check_resume


def trial_seed(base: int, problem_id: str, attempt: int) -> int:
    """Stable per-trial seed, independent of scheduling and Python hash salt."""
    if type(base) is not int or base < 0:
        raise ValueError("seed_base must be a nonnegative integer")
    identity = json.dumps([base, problem_id, attempt], separators=(",", ":"))
    return int.from_bytes(sha256(identity.encode()).digest()[:4], "big") & 0x7FFFFFFF


async def run_trial(
    problem: ProblemSpec,
    attempt: int,
    directory: Path,
    client: ChatClient,
    agent: AgentConfig,
    *,
    resume: bool = False,
) -> str:
    """Generate and statically validate one problem attempt."""
    sampling = Sampling(agent.model_name, **dict(agent.kwargs.get("sampling", {})))
    seed_base = agent.kwargs.get("seed_base")
    if seed_base is not None:
        if sampling.seed is not None:
            raise ValueError("set seed_base or sampling.seed, not both")
        sampling = replace(sampling, seed=trial_seed(seed_base, problem.id, attempt))
    style: PromptStyle = agent.kwargs.get("style", "one-step")
    client_name = agent.kwargs.get("client_name", "vllm")
    agent_dir = directory / "agent"
    artifacts = directory / "artifacts"
    validation = directory / "validation"
    for path in (agent_dir, artifacts, validation):
        path.mkdir(parents=True, exist_ok=resume)
    started_at = datetime.now(UTC).isoformat()
    print(
        json.dumps(
            {
                "event": "trial_started",
                "trial": directory.name,
                "at": started_at,
                "seed": sampling.seed,
            }
        ),
        flush=True,
    )

    try:
        response, _ = await generate(
            client,
            problem,
            style,
            sampling,
            formatting_max_tokens=agent.kwargs.get("formatting_max_tokens"),
            checkpoint_dir=agent_dir,
            require_complete_stages=agent.kwargs.get("require_complete_stages", False),
            resume=resume,
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
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "agent_info": {
            "name": "critpt",
            "model_info": {"provider": client_name, "name": sampling.model},
        },
        "config": {
            "agent": {
                "kwargs": {
                    "client_name": client_name,
                    "request_profile": agent.kwargs.get("request_profile"),
                    "style": style,
                    "sampling": asdict(sampling),
                    "seed_base": seed_base,
                    "formatting_max_tokens": agent.kwargs.get("formatting_max_tokens"),
                    "context_window": agent.kwargs.get("context_window"),
                    "require_complete_stages": agent.kwargs.get(
                        "require_complete_stages", False
                    ),
                }
            }
        },
        "static_result": result,
    }
    (directory / "result.json").write_text(
        json.dumps(trial, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"event": "trial_finished", "trial": directory.name, **result}),
        flush=True,
    )
    return result["status"]


async def run_job(
    path: Path,
    client: ChatClient | None = None,
    *,
    task_name: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run every task in one Harbor job config without code execution."""
    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = JobConfig.model_validate(raw_config)
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
    if resume:
        saved_path = output / "job-config.yaml"
        previous = JobConfig.model_validate(
            yaml.safe_load(saved_path.read_text(encoding="utf-8"))
        )
        check_resume(previous, config, raw_config.get("resume_migration"))
    output.mkdir(parents=True, exist_ok=resume)
    kwargs = agent.kwargs
    client_name = kwargs.get("client_name", "vllm")
    if client_name not in CLIENTS:
        raise ValueError(f"unknown client {client_name!r}")
    sampling = Sampling(agent.model_name, **dict(kwargs.get("sampling", {})))
    if kwargs.get("seed_base") is not None:
        if client_name not in {"vllm", "openai", "aliyun"}:
            raise ValueError("seed_base requires a vLLM or OpenAI-compatible endpoint")
        if sampling.seed is not None:
            raise ValueError("set seed_base or sampling.seed, not both")
        trial_seed(kwargs["seed_base"], ids[0], 0)
    client_options = {}
    if "request_profile" in kwargs:
        if client_name != "aliyun":
            raise ValueError("request_profile requires client_name: aliyun")
        client_options["request_profile"] = kwargs["request_profile"]
    if kwargs.get("context_window") is not None:
        if client_name != "vllm":
            raise ValueError(
                "dynamic context budgeting requires the vLLM tokenizer endpoint"
            )
        client_options = {
            "context_window": kwargs["context_window"],
            "context_safety_tokens": kwargs.get("context_safety_tokens", 32),
        }
    connection = (
        nullcontext(client)
        if client is not None
        else CLIENTS[client_name](
            kwargs.get("base_url", "http://127.0.0.1:8000/v1"),
            sampling,
            stream=bool(kwargs.get("stream", False)),
            api_key=_api_key(kwargs.get("api_key_env")),
            timeout=kwargs.get("timeout", 1200),
            **client_options,
        )
    )

    work = [
        (problem, attempt)
        for attempt in range(config.n_attempts)
        for problem in problems
    ]
    # Refill a slot when a trial finishes or fails, without waiting for unrelated
    # slower trials in the same batch. Its two stages remain sequential.
    slots = asyncio.Semaphore(config.n_concurrent_trials)

    async def scheduled(problem, attempt, active):
        async with slots:
            directory = output / f"{problem.id}__attempt-{attempt}"
            completed = directory / "result.json"
            if resume and completed.exists():
                result = json.loads(completed.read_text(encoding="utf-8"))
                if (
                    result.get("task_name") != f"critpt/{problem.id}"
                    or result.get("attempt") != attempt
                ):
                    raise ValueError(f"invalid completed trial identity: {directory}")
                return result["static_result"]["status"]
            return await run_trial(
                problem,
                attempt,
                directory,
                active,
                agent,
                **({"resume": True} if resume else {}),
            )

    config_name = (
        "resume-config-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".yaml"
        if resume
        else "job-config.yaml"
    )
    (output / config_name).write_text(
        path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    async with connection as active:
        await active.preflight()
        statuses = await asyncio.gather(*(scheduled(p, a, active) for p, a in work))
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
