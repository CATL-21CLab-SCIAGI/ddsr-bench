from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench.training.schemas import Generation, SftSample, Trajectory
from ddsr_bench.training.sft import native_samples

REPOSITORY_COMMIT = "e14e0ca3d3db0f02928a18d435b9c3d15656a7c5"


def _generations(response: dict[str, Any]) -> tuple[Generation, ...]:
    calls = response.get("calls")
    if not isinstance(calls, list) or not calls:
        raise TypeError("SciBench response requires model calls")
    strategy = response.get("strategy")
    names = ("derivation", "answer") if strategy == "zero-cot" else ("answer",)
    if len(calls) != len(names):
        raise ValueError("SciBench calls do not match its strategy")
    generations = []
    for name, call in zip(names, calls, strict=True):
        messages, recorded = call.get("messages"), call.get("response")
        if not isinstance(messages, list) or not isinstance(recorded, dict):
            raise TypeError("SciBench call requires messages and a response")
        prompt = tuple(
            {"role": message["role"], "content": message["content"]}
            for message in messages
        )
        completion: dict[str, Any] = {
            "role": "assistant",
            "content": recorded.get("content"),
        }
        if isinstance(recorded.get("reasoning"), str) and recorded["reasoning"]:
            completion["reasoning"] = recorded["reasoning"]
        generations.append(Generation(name, prompt, completion, "model", quality={}))
    return tuple(generations)


def normalize(
    trial: Path,
    response: dict[str, Any],
    result: dict[str, Any],
    validation: dict[str, Any],
) -> Trajectory:
    problem_id, problem = response.get("problem_id"), response.get("problem")
    if not isinstance(problem_id, str) or not isinstance(problem, dict):
        raise TypeError("SciBench response requires public problem identity")
    generations = _generations(response)
    kwargs = ((result.get("config") or {}).get("agent") or {}).get("kwargs") or {}
    response_path = trial / "agent" / "response.json"
    encoded = json.dumps(response["calls"], ensure_ascii=False, sort_keys=True)
    return Trajectory(
        schema_version=1,
        id=result.get("trial_name") or trial.name,
        benchmark="scibench",
        problem_id=problem_id,
        generations=generations,
        teacher={
            "client": kwargs.get("client_name"),
            "model": response.get("model"),
            "strategy": response.get("strategy"),
            "sampling": response.get("sampling"),
        },
        quality={
            "finish_reasons": [
                call["response"].get("finish_reason") for call in response["calls"]
            ],
            "validation": validation.get("mode"),
            "status": validation.get("status"),
            "reward": validation.get("reward"),
            "verified": validation.get("mode") == "scibench"
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
        raise ValueError(f"{view!r} view is unavailable for SciBench")
    return native_samples(trajectory)
