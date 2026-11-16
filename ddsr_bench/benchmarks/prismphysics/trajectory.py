from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.prismphysics.data.loader import REVISION
from ddsr_bench.training.schemas import Generation, SftSample, Trajectory
from ddsr_bench.training.sft import native_samples

SCORER_COMMIT = "d2958d12a99e7a38a41832e4e0cc9f40ef7d474f"
FORMULA_ENGINE = "gradePhyX-1.0.2-adapted"


def _generation(response: dict[str, Any]) -> Generation:
    messages, responses = response.get("messages"), response.get("responses")
    if not isinstance(messages, list) or len(messages) != 2:
        raise TypeError("PRISM-Physics requires one user-assistant conversation")
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != ["user", "assistant"]:
        raise ValueError("PRISM-Physics response has an invalid conversation")
    if not isinstance(responses, list) or len(responses) != 1:
        raise TypeError("PRISM-Physics response requires exactly one model response")
    contents = [message.get("content") for message in messages]
    if any(not isinstance(content, str) for content in contents):
        raise TypeError("PRISM-Physics messages require text content")
    recorded = responses[0]
    if not isinstance(recorded, dict) or recorded.get("content") != contents[-1]:
        raise ValueError("PRISM-Physics message and response content do not match")
    completion = {"role": "assistant", "content": contents[-1]}
    reasoning = recorded.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        completion["reasoning"] = reasoning
    prompt = ({"role": "user", "content": contents[0]},)
    return Generation("answer", prompt, completion, "model", quality={})


def _problem(response: dict[str, Any]) -> dict[str, Any]:
    problem = response.get("problem")
    if not isinstance(problem, dict):
        raise TypeError("PRISM-Physics response requires public problem metadata")
    required = ("raw_id", "shard", "domain", "source", "images")
    if any(field not in problem for field in required):
        raise TypeError("PRISM-Physics public problem metadata is incomplete")
    return {field: problem[field] for field in required}


def normalize(
    trial: Path,
    response: dict[str, Any],
    result: dict[str, Any],
    validation: dict[str, Any],
) -> Trajectory:
    """Normalize one PRISM trial without exposing the reference formula DAG."""
    problem_id = response.get("problem_id")
    responses = response.get("responses")
    if not isinstance(problem_id, str) or not problem_id:
        raise TypeError("PRISM-Physics response requires a problem ID")
    if not isinstance(responses, list) or not responses:
        raise TypeError("PRISM-Physics response requires a model response")
    problem = _problem(response)
    generation = _generation(response)
    kwargs = ((result.get("config") or {}).get("agent") or {}).get("kwargs") or {}
    response_path = trial / "agent" / "response.json"
    encoded = json.dumps(generation.prompt + (generation.completion,), sort_keys=True)
    mode, status = validation.get("mode"), validation.get("status")
    return Trajectory(
        schema_version=1,
        id=result.get("trial_name") or trial.name,
        benchmark="prismphysics",
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
            "status": status,
            "reward": validation.get("reward"),
            "verified": mode == "prismphysics" and status == "passed",
            "parsed": validation.get("success") is True,
            "threshold": validation.get("threshold"),
            "direct_matches": validation.get("direct_matches"),
            "credited": validation.get("credited"),
        },
        metadata=problem,
        provenance={
            "source_response": str(response_path.relative_to(trial.parent)),
            "dataset_revision": REVISION,
            "scorer_commit": SCORER_COMMIT,
            "formula_engine": FORMULA_ENGINE,
            "content_sha256": sha256(encoded.encode()).hexdigest(),
            "response_sha256": sha256(response_path.read_bytes()).hexdigest(),
        },
    )


def sft_samples(trajectory: Trajectory, view: str) -> tuple[SftSample, ...]:
    """Export the single recorded PRISM-Physics model call."""
    if view not in ("full", "native"):
        raise ValueError(f"{view!r} view is unavailable for PRISM-Physics")
    return native_samples(trajectory)
