from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.scibench.data.schemas import ProblemSpec, Strategy
from ddsr_bench.benchmarks.scibench.generation.prompts import (
    initial_messages,
    zero_messages,
)
from ddsr_bench.generation.client import ChatClient, Sampling


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(encoded.encode()).hexdigest()


async def generate(
    problem: ProblemSpec,
    strategy: Strategy,
    client: ChatClient,
    sampling: Sampling,
    stream_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    requests = [initial_messages(problem, strategy)]
    responses = [await client.chat(requests[0], stream_path=stream_path)]
    if strategy == "zero-cot":
        requests.append(zero_messages(problem, responses[0].content))
        responses.append(await client.chat(requests[1], stream_path=stream_path))
    return responses[-1].content, {
        "problem_id": problem.id,
        "problem": {"split": problem.split, "textbook": problem.textbook},
        "strategy": strategy,
        "model": responses[-1].model,
        "seed": sampling.seed,
        "sampling": asdict(sampling),
        "calls": [
            {
                "messages": [
                    message | {"sha256": _hash(message["content"])}
                    for message in request
                ],
                "response": asdict(response) | {"sha256": _hash(response.content)},
            }
            for request, response in zip(requests, responses, strict=True)
        ],
    }
