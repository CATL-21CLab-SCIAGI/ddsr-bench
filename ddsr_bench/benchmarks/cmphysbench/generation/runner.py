from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from typing import Any

from ddsr_bench.benchmarks.cmphysbench.data.schemas import ProblemSpec
from ddsr_bench.benchmarks.cmphysbench.generation.prompts import messages
from ddsr_bench.generation.client import ChatClient, Sampling


def _hash(content: str) -> str:
    return sha256(content.encode()).hexdigest()


async def generate(
    problem: ProblemSpec,
    client: ChatClient,
    sampling: Sampling,
) -> tuple[str, dict[str, Any]]:
    """Generate one response and return its public audit record."""
    request = messages(problem)
    response = await client.chat(request)
    conversation = (*request, {"role": "assistant", "content": response.content})
    recorded_messages = [
        message | {"sha256": _hash(message["content"])} for message in conversation
    ]
    recorded_response = asdict(response) | {"sha256": _hash(response.content)}
    return response.content, {
        "problem_id": problem.id,
        "problem": {
            "context": problem.context,
            "question": problem.question,
            "symbols": problem.symbols,
            "answer_type": problem.answer_type,
            "topic": problem.topic,
        },
        "strategy": "single-turn",
        "model": response.model,
        "seed": sampling.seed,
        "sampling": asdict(sampling),
        "messages": recorded_messages,
        "responses": [recorded_response],
    }
