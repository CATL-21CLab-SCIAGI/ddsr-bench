from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench.training.schemas import Generation, SftSample, Trajectory
from ddsr_bench.training.sft import native_samples

UPSTREAM_COMMIT = "e3158ea011d4235245a547460d3688d7ccbf9900"


def _generation(step: dict[str, Any], quality: dict[str, Any]) -> Generation:
    step_id = step.get("id")
    source = step.get("source")
    code = step.get("code")
    if not isinstance(step_id, str) or not isinstance(code, str):
        raise TypeError("SciCode step requires an ID and generated code")
    if source == "fixed":
        return Generation(
            step_id,
            prompt=(),
            completion={"role": "assistant", "content": code},
            source="fixed",
            quality=quality,
        )
    if source != "model":
        raise ValueError(f"SciCode step {step_id!r} has unknown source {source!r}")

    prompt = step.get("prompt")
    response = step.get("response")
    if not isinstance(prompt, str) or not isinstance(response, dict):
        raise TypeError(f"SciCode model step {step_id!r} requires a response")
    content = response.get("content")
    if not isinstance(content, str):
        raise TypeError(f"SciCode model step {step_id!r} requires response content")
    completion = {"role": "assistant", "content": content}
    reasoning = response.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        completion["reasoning"] = reasoning
    return Generation(
        step_id,
        prompt=({"role": "user", "content": prompt},),
        completion=completion,
        source="model",
        quality=quality,
    )


def normalize(
    trial: Path,
    response: dict[str, Any],
    result: dict[str, Any],
    validation: dict[str, Any],
) -> Trajectory:
    """Normalize one SciCode trial without exposing tests or target data."""
    problem_id = response.get("problem_id")
    steps = response.get("steps")
    if not isinstance(problem_id, str) or not problem_id:
        raise TypeError("response record requires a problem ID")
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise TypeError("SciCode response record requires step objects")

    verified_steps = validation.get("steps") or []
    if not isinstance(verified_steps, list):
        raise TypeError("SciCode validation steps must be a list")
    step_quality = {
        item["id"]: {"status": item.get("status")}
        for item in verified_steps
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    generations = tuple(
        _generation(step, step_quality.get(step.get("id"), {})) for step in steps
    )
    if not generations:
        raise ValueError("SciCode response record has no steps")

    agent = (result.get("config") or {}).get("agent") or {}
    kwargs = agent.get("kwargs") or {}
    model_responses = [
        step["response"]
        for step in steps
        if step.get("source") == "model" and isinstance(step.get("response"), dict)
    ]
    model = next(
        (
            item.get("model")
            for item in model_responses
            if isinstance(item.get("model"), str)
        ),
        agent.get("model_name"),
    )
    mode, reward = validation.get("mode"), validation.get("reward")
    response_path = trial / "agent" / "response.json"
    encoded = json.dumps(
        [asdict(generation) for generation in generations],
        ensure_ascii=False,
        sort_keys=True,
    ).encode()
    return Trajectory(
        schema_version=2,
        id=result.get("trial_name") or trial.name,
        benchmark="scicode",
        problem_id=problem_id,
        generations=generations,
        teacher={
            "client": kwargs.get("client_name"),
            "model": model,
            "strategy": "sequential",
            "sampling": kwargs.get("sampling"),
        },
        quality={
            "finish_reasons": [item.get("finish_reason") for item in model_responses],
            "validation": mode,
            "status": validation.get("status"),
            "reward": reward,
            "verified": mode == "scicode" and reward == 1,
        },
        metadata={
            "with_background": response.get("with_background"),
            "steps": [generation.id for generation in generations],
        },
        provenance={
            "source_response": str(response_path.relative_to(trial.parent)),
            "upstream_commit": UPSTREAM_COMMIT,
            "content_sha256": sha256(encoded).hexdigest(),
            "response_sha256": sha256(response_path.read_bytes()).hexdigest(),
        },
    )


def sft_samples(trajectory: Trajectory, view: str) -> tuple[SftSample, ...]:
    """Export model-generated SciCode steps; fixed steps are context only."""
    if view not in ("full", "native"):
        raise ValueError(f"{view!r} view is unavailable for SciCode")
    return native_samples(trajectory)
