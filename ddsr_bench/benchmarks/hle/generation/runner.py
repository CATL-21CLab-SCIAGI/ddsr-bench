from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.hle.data.schemas import ProblemSpec
from ddsr_bench.benchmarks.hle.generation.prompts import messages
from ddsr_bench.generation.client import ChatClient, Sampling


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(encoded.encode()).hexdigest()


async def generate(
    problem: ProblemSpec,
    client: ChatClient,
    sampling: Sampling,
    stream_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate one HLE response and return its public audit record."""
    request = messages(problem, sampling.model)
    response = await client.chat(request, stream_path=stream_path)
    conversation = (*request, {"role": "assistant", "content": response.content})
    recorded_messages = [
        message | {"sha256": _hash(message["content"])} for message in conversation
    ]
    recorded_response = asdict(response) | {"sha256": _hash(response.content)}
    return response.content, {
        "problem_id": problem.id,
        "problem": {
            "subset": problem.subset,
            "category": problem.category,
            "subject": problem.subject,
            "question": problem.question,
            "image": problem.image is not None,
        },
        "strategy": "single-turn",
        "model": response.model,
        "seed": sampling.seed,
        "sampling": asdict(sampling),
        "messages": recorded_messages,
        "responses": [recorded_response],
    }
