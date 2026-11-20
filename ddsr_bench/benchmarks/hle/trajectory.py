from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.hle.data.loader import REVISION
from ddsr_bench.training.schemas import Generation, SftSample, Trajectory
from ddsr_bench.training.sft import native_samples

HARNESS_COMMIT = "22ed3074b1e7b134bcbc09028d0ba320839b0655"
VERIFIED_COMMIT = "b705e0fb541c025a1532ce0d60d70ae2f53b00e0"


def _generation(response: dict[str, Any]) -> Generation:
    messages = response.get("messages")
    responses = response.get("responses")
    if not isinstance(messages, list) or len(messages) != 3:
        raise TypeError("HLE response requires one three-message conversation")
    if messages[0].get("role") not in ("system", "user") or [
        message.get("role") for message in messages[1:]
    ] != ["user", "assistant"]:
        raise ValueError("HLE response has an invalid conversation")
    if not isinstance(responses, list) or len(responses) != 1:
        raise TypeError("HLE response requires exactly one model response")
    recorded = responses[0]
    if not isinstance(recorded, dict) or recorded.get("content") != messages[-1].get(
        "content"
    ):
        raise ValueError("HLE message and response content do not match")
    completion: dict[str, Any] = {
        "role": "assistant",
        "content": messages[-1]["content"],
    }
    reasoning = recorded.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        completion["reasoning"] = reasoning
    prompt = tuple(
        {"role": message["role"], "content": message["content"]}
        for message in messages[:2]
    )
    return Generation("answer", prompt, completion, "model", quality={})


def normalize(
    trial: Path,
    response: dict[str, Any],
    result: dict[str, Any],
    validation: dict[str, Any],
) -> Trajectory:
    """Normalize one HLE trial without exposing answer or rationale."""
    problem_id = response.get("problem_id")
    problem = response.get("problem")
    responses = response.get("responses")
    if not isinstance(problem_id, str) or not problem_id:
        raise TypeError("HLE response requires a problem ID")
    if not isinstance(problem, dict):
        raise TypeError("HLE response requires public problem fields")
    generation = _generation(response)
    kwargs = ((result.get("config") or {}).get("agent") or {}).get("kwargs") or {}
    response_path = trial / "agent" / "response.json"
    encoded = json.dumps(
        generation.prompt + (generation.completion,),
        ensure_ascii=False,
        sort_keys=True,
    )
    return Trajectory(
        schema_version=1,
        id=result.get("trial_name") or trial.name,
        benchmark="hle",
        problem_id=problem_id,
        generations=(generation,),
        teacher={
            "client": kwargs.get("client_name"),
            "model": response.get("model"),
            "strategy": response.get("strategy"),
            "sampling": response.get("sampling"),
        },
        quality={
            "finish_reasons": [responses[0].get("finish_reason")],
            "validation": validation.get("mode"),
            "status": validation.get("status"),
            "reward": validation.get("reward"),
            "verified": validation.get("mode") == "hle"
            and validation.get("reward") == 1,
            "judge_model": validation.get("judge_model"),
        },
        metadata=problem,
        provenance={
            "source_response": str(response_path.relative_to(trial.parent)),
            "dataset_revision": REVISION,
            "harness_commit": HARNESS_COMMIT,
            "verified_commit": VERIFIED_COMMIT,
            "content_sha256": sha256(encoded.encode()).hexdigest(),
            "response_sha256": sha256(response_path.read_bytes()).hexdigest(),
        },
    )


def sft_samples(trajectory: Trajectory, view: str) -> tuple[SftSample, ...]:
    """Export the single recorded HLE model call, including image parts."""
    if view not in ("full", "native"):
        raise ValueError(f"{view!r} view is unavailable for HLE")
    return native_samples(trajectory)
