"""Run static benchmark evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import yaml

from ddsr_bench.benchmarks.critpt.data.loader import load_challenge
from ddsr_bench.benchmarks.critpt.data.schemas import ProblemSpec
from ddsr_bench.benchmarks.critpt.generation.runner import generate
from ddsr_bench.benchmarks.registry import static_runner
from ddsr_bench.generation.client import CLIENTS, Sampling
from ddsr_bench.grading.validation import extract_answer

from .utils import read_api_key


async def run_job(path: Path, task_name: str | None = None) -> dict[str, Any]:
    """Dispatch one static job to its benchmark adapter."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("job configuration must be a mapping")
    name = config.get("benchmark", "critpt")
    if not isinstance(name, str):
        raise TypeError("job benchmark must be a name")
    return await static_runner(name)(path, task_name=task_name)


def _select(path: Path, task_name: str | None) -> ProblemSpec:
    challenge = load_challenge(path)
    if task_name is None:
        return challenge.main.spec
    for problem in challenge.problems:
        if problem.spec.id == task_name:
            return problem.spec
    raise ValueError(f"task {task_name!r} not found in {challenge.id!r}")


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    problem = _select(args.challenge, args.include_task_name)
    openai = args.client in ("openai", "bedrock")
    sampling = Sampling(
        args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        reasoning_effort=(
            "none" if args.no_thinking and openai else args.reasoning_effort
        ),
        enable_thinking=False if args.no_thinking and not openai else None,
    )
    async with CLIENTS[args.client](
        args.base_url,
        sampling,
        stream=args.stream,
        api_key=read_api_key(args.api_key_env),
    ) as client:
        await client.preflight()
        response, record = await generate(client, problem, args.style, sampling)

    args.output.mkdir(parents=True)
    (args.output / "response.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        code = extract_answer(response, problem.code_template)
        (args.output / "answer.py").write_text(code, encoding="utf-8")
        result: dict[str, Any] = {
            "reward": None,
            "mode": "static",
            "status": "validated",
        }
    except Exception as error:  # noqa: BLE001 - preserve failed model attempts
        result = {
            "reward": None,
            "mode": "static",
            "status": "error",
            "error": type(error).__name__,
            "message": str(error),
        }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run static benchmark evaluation or solve one CritPt problem"
    )
    parser.add_argument("challenge", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--include-task-name")
    parser.add_argument("--style", choices=("one-step", "two-step"), default="one-step")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key-env")
    parser.add_argument("--client", choices=tuple(CLIENTS), default="vllm")
    parser.add_argument("--model", default="ddsr-local")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
    )
    thinking.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    if args.config:
        if args.challenge or args.output:
            parser.error("--config cannot be combined with a challenge or output")
        result = asyncio.run(run_job(args.config, args.include_task_name))
    else:
        if args.challenge is None or args.output is None:
            parser.error("challenge and output are required without --config")
        result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2))
