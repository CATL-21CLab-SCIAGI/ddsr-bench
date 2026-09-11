from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench.training.schemas import Generation, SftSample, Trajectory
from ddsr_bench.training.sft import native_samples

UPSTREAM_COMMIT = "b2cd8571279450f0861759f47d98e9fc577aa993"
_PUBLIC_FIELDS = ("context", "question", "symbols", "answer_type", "topic")


def _generation(response: dict[str, Any]) -> Generation:
    messages = response.get("messages")
    responses = response.get("responses")
    if not isinstance(messages, list) or len(messages) != 3:
        raise TypeError("CMPhysBench response requires one three-message conversation")
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != ["system", "user", "assistant"]:
        raise ValueError("CMPhysBench response has an invalid conversation")
    if not isinstance(responses, list) or len(responses) != 1:
        raise TypeError("CMPhysBench response requires exactly one model response")
    contents = [message.get("content") for message in messages]
    if any(not isinstance(content, str) for content in contents):
        raise TypeError("CMPhysBench messages require text content")
    recorded = responses[0]
    if not isinstance(recorded, dict) or recorded.get("content") != contents[-1]:
        raise ValueError("CMPhysBench message and response content do not match")
    completion = {"role": "assistant", "content": contents[-1]}
    reasoning = recorded.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        completion["reasoning"] = reasoning
    prompt = tuple(
        {"role": messages[index]["role"], "content": contents[index]}
        for index in range(2)
    )
    return Generation("answer", prompt, completion, "model", quality={})


def _problem(response: dict[str, Any]) -> dict[str, str]:
    problem = response.get("problem")
    if not isinstance(problem, dict):
        raise TypeError("CMPhysBench response requires public problem fields")
    public = {field: problem.get(field) for field in _PUBLIC_FIELDS}
    if any(not isinstance(value, str) for value in public.values()):
        raise TypeError("CMPhysBench public problem fields must contain text")
    return public


def normalize(
    trial: Path,
    response: dict[str, Any],
    result: dict[str, Any],
    validation: dict[str, Any],
) -> Trajectory:
    """Normalize one CMPhysBench trial without exposing its reference answer."""
    problem_id = response.get("problem_id")
    responses = response.get("responses")
    if not isinstance(problem_id, str) or not problem_id:
        raise TypeError("CMPhysBench response requires a problem ID")
    problem = _problem(response)
    generation = _generation(response)
    kwargs = ((result.get("config") or {}).get("agent") or {}).get("kwargs") or {}
    response_path = trial / "agent" / "response.json"
    encoded = json.dumps(generation.prompt + (generation.completion,), sort_keys=True)
    mode, reward = validation.get("mode"), validation.get("reward")
    return Trajectory(
        schema_version=2,
        id=result.get("trial_name") or trial.name,
        benchmark="cmphysbench",
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
            "validation": mode,
            "status": validation.get("status"),
            "reward": reward,
            "verified": mode == "seed" and reward == 1,
            "seed_score": validation.get("seed_score"),
        },
        metadata=problem,
        provenance={
            "source_response": str(response_path.relative_to(trial.parent)),
            "upstream_commit": UPSTREAM_COMMIT,
            "content_sha256": sha256(encoded.encode()).hexdigest(),
            "response_sha256": sha256(response_path.read_bytes()).hexdigest(),
        },
    )


def sft_samples(trajectory: Trajectory, view: str) -> tuple[SftSample, ...]:
    """Export the single recorded CMPhysBench model call."""
    if view not in ("full", "native"):
        raise ValueError(f"{view!r} view is unavailable for CMPhysBench")
    return native_samples(trajectory)
