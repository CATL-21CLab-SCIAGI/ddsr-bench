from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.mmmupro.data.schemas import Mode, ProblemSpec
from ddsr_bench.benchmarks.mmmupro.generation.prompts import messages
from ddsr_bench.generation.client import ChatClient, Sampling


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(encoded.encode()).hexdigest()


async def generate(
    problem: ProblemSpec,
    mode: Mode,
    client: ChatClient,
    sampling: Sampling,
    stream_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    request = messages(problem, mode)
    response = await client.chat(request, stream_path=stream_path)
    conversation = (*request, {"role": "assistant", "content": response.content})
    return response.content, {
        "problem_id": problem.id,
        "problem": {
            "setting": problem.setting,
            "subject": problem.subject,
            "domain": problem.domain,
            "mode": mode,
        },
        "strategy": mode,
        "model": response.model,
        "seed": sampling.seed,
        "sampling": asdict(sampling),
        "messages": [
            message | {"sha256": _hash(message["content"])} for message in conversation
        ],
        "responses": [asdict(response) | {"sha256": _hash(response.content)}],
    }
