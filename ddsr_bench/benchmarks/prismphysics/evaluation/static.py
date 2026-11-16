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

from ddsr_bench.benchmarks.prismphysics.data.loader import (
    THRESHOLD,
    _validate_dag,
    load_split,
)
from ddsr_bench.benchmarks.prismphysics.data.schemas import (
    AnswerSpec,
    FormulaNode,
    Problem,
)
from ddsr_bench.benchmarks.prismphysics.evaluation.prepare import decode_instruction
from ddsr_bench.benchmarks.prismphysics.evaluation.verifier import verify
from ddsr_bench.benchmarks.prismphysics.generation.runner import generate
from ddsr_bench.benchmarks.utils import benchmark_config, read_json, write_json
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
    threshold: float = THRESHOLD,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate and score one PRISM-Physics problem attempt locally."""
    if problem.answer is None:
        raise ValueError(f"PRISM-Physics problem {problem.spec.id!r} has no reference")
    if resume:
        saved = completed(directory, f"prismphysics/{problem.spec.id}", attempt)
        if saved is not None:
            return saved

    started_at = datetime.now(UTC).isoformat()
    directory.mkdir(parents=True)
    agent = directory / "agent"
    artifacts = directory / "artifacts"
    validation = directory / "validation"
    for path in (agent, artifacts, validation):
        path.mkdir()
    try:
        response, record = await generate(problem.spec, client, sampling)
        write_json(agent / "response.json", record)
        (artifacts / "answer.txt").write_text(response, encoding="utf-8")
        result = await asyncio.to_thread(
            verify, response, problem.answer, threshold, timeout_sec
        )
    except Exception as error:  # noqa: BLE001 - one failed attempt receives zero
        result = {
            "reward": 0.0,
            "mode": "prismphysics",
            "status": "error",
            "error": type(error).__name__,
        }

    result.update(domain=problem.spec.domain, shard=problem.spec.shard)
    write_json(validation / "result.json", result)
    trial = {
        "task_name": f"prismphysics/{problem.spec.id}",
        "trial_name": directory.name,
        "attempt": attempt,
        "started_at": started_at,
        "agent_info": {
            "name": "prismphysics",
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


def _answer(reference: dict[str, Any], problem_id: str, threshold: float) -> AnswerSpec:
    if reference.get("threshold") != threshold:
        raise ValueError(f"invalid prepared threshold for {problem_id!r}")
    formulas = reference.get("formulas")
    if not isinstance(formulas, list) or any(
        not isinstance(node, dict) for node in formulas
    ):
        raise TypeError(f"invalid prepared reference for {problem_id!r}")
    try:
        nodes = tuple(
            FormulaNode(
                index=node["index"],
                formula=node["formula"],
                dependencies=tuple(node["dependencies"]),
                final=node["final"],
            )
            for node in formulas
        )
    except (KeyError, TypeError) as error:
        raise TypeError(f"invalid prepared reference for {problem_id!r}") from error
    if any(
        type(node.index) is not int
        or not isinstance(node.formula, str)
        or type(node.final) is not bool
        or any(type(dependency) is not int for dependency in node.dependencies)
        for node in nodes
    ):
        raise TypeError(f"invalid prepared reference for {problem_id!r}")
    _validate_dag(nodes, problem_id)
    return AnswerSpec(nodes)


def _prepared(config: JobConfig, threshold: float) -> list[Problem]:
    problems = []
    for dataset in config.datasets:
        if dataset.path is None:
            continue
        for instruction in sorted(dataset.path.glob("*/instruction.md")):
            spec = decode_instruction(instruction.read_text(encoding="utf-8"))
            reference = read_json(instruction.parent / "tests" / "reference.json")
            problems.append(Problem(spec, _answer(reference, spec.id, threshold)))
    if not problems:
        raise ValueError("no prepared PRISM-Physics tasks found")
    return problems


async def run_job(
    path: Path,
    client: ChatClient | None = None,
    *,
    task_name: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run a PRISM-Physics static job from pinned data or prepared tasks."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("benchmark") != "prismphysics":
        raise ValueError("job benchmark must be 'prismphysics'")
    settings = benchmark_config("prismphysics")

    config = JobConfig.model_validate(raw)
    if len(config.agents) != 1:
        raise ValueError("PRISM-Physics jobs require exactly one agent")
    agent = config.agents[0]
    expected = "ddsr_bench.benchmarks.prismphysics.evaluation.harbor:PRISMPhysicsAgent"
    if agent.name != expected or not agent.model_name:
        raise ValueError("PRISM-Physics job requires its named agent and model")

    source = "huggingface"
    try:
        problems = load_split(str(settings["revision"]))
    except (ImportError, OSError, ValueError) as source_error:
        try:
            problems = _prepared(config, float(settings["threshold"]))
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
                            threshold=float(settings["threshold"]),
                            resume=resume,
                        )
                        for problem, attempt in group
                    )
                )
            )
    scored = [
        result for result in results if result["status"] in ("passed", "different")
    ]
    return {
        "output": str(output),
        "source": source,
        "trials": len(results),
        "scored": len(scored),
        "passed": sum(result["status"] == "passed" for result in results),
        "mean_score": (
            sum(result["reward"] for result in scored) / len(scored) if scored else 0.0
        ),
        "errors": sum(result["status"] in ("error", "timeout") for result in results),
    }
