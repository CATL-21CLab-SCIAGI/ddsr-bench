from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench.training.schemas import Generation, SftSample, Trajectory
from ddsr_bench.training.sft import native_samples

REPOSITORY_COMMIT = "a9af8bf354887043bd280a617c8716ef54e7aaec"


def _generation(response: dict[str, Any]) -> Generation:
    messages, responses = response.get("messages"), response.get("responses")
    if not isinstance(messages, list) or len(messages) != 3:
        raise TypeError("JEEBench response requires three conversation messages")
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("JEEBench response has invalid message roles")
    if not isinstance(responses, list) or len(responses) != 1:
        raise TypeError("JEEBench response requires one model response")
    recorded = responses[0]
    if not isinstance(recorded, dict) or recorded.get("content") != messages[-1].get(
        "content"
    ):
        raise ValueError("JEEBench response content does not match its message")
    completion: dict[str, Any] = {
        "role": "assistant",
        "content": messages[-1]["content"],
    }
    if isinstance(recorded.get("reasoning"), str) and recorded["reasoning"]:
        completion["reasoning"] = recorded["reasoning"]
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
    problem_id, problem = response.get("problem_id"), response.get("problem")
    responses = response.get("responses")
    if not isinstance(problem_id, str) or not isinstance(problem, dict):
        raise TypeError("JEEBench response requires public problem identity")
    generation = _generation(response)
    kwargs = ((result.get("config") or {}).get("agent") or {}).get("kwargs") or {}
    response_path = trial / "agent" / "response.json"
    encoded = json.dumps(
        generation.prompt + (generation.completion,), ensure_ascii=False, sort_keys=True
    )
    return Trajectory(
        schema_version=1,
        id=result.get("trial_name") or trial.name,
        benchmark="jeebench",
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
            "verified": validation.get("mode") == "jeebench"
            and validation.get("reward") == 1,
        },
        metadata=problem,
        provenance={
            "source_response": str(response_path.relative_to(trial.parent)),
            "repository_commit": REPOSITORY_COMMIT,
            "content_sha256": sha256(encoded.encode()).hexdigest(),
            "response_sha256": sha256(response_path.read_bytes()).hexdigest(),
        },
    )


def sft_samples(trajectory: Trajectory, view: str) -> tuple[SftSample, ...]:
    if view not in ("full", "native"):
        raise ValueError(f"{view!r} view is unavailable for JEEBench")
    return native_samples(trajectory)
