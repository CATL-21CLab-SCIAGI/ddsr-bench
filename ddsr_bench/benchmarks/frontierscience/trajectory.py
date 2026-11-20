from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.frontierscience.data.loader import REVISION
from ddsr_bench.benchmarks.utils import benchmark_config
from ddsr_bench.training.schemas import Generation, SftSample, Trajectory
from ddsr_bench.training.sft import native_samples

HARNESS_COMMIT = str(benchmark_config("frontierscience")["harness_revision"])


def _generation(response: dict[str, Any]) -> Generation:
    messages = response.get("messages")
    responses = response.get("responses")
    if not isinstance(messages, list) or len(messages) != 2:
        raise TypeError("FrontierScience response requires one user/assistant turn")
    if [message.get("role") for message in messages] != ["user", "assistant"]:
        raise ValueError("FrontierScience response has an invalid conversation")
    if not isinstance(responses, list) or len(responses) != 1:
        raise TypeError("FrontierScience response requires exactly one model response")
    recorded = responses[0]
    if not isinstance(recorded, dict) or recorded.get("content") != messages[-1].get(
        "content"
    ):
        raise ValueError("FrontierScience message and response do not match")
    completion: dict[str, Any] = {
        "role": "assistant",
        "content": messages[-1]["content"],
    }
    reasoning = recorded.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        completion["reasoning"] = reasoning
    prompt = ({"role": "user", "content": messages[0]["content"]},)
    return Generation("answer", prompt, completion, "model", quality={})


def normalize(
    trial: Path,
    response: dict[str, Any],
    result: dict[str, Any],
    validation: dict[str, Any],
) -> Trajectory:
    """Normalize one attempt without exposing its answer or rubric."""
    problem_id = response.get("problem_id")
    problem = response.get("problem")
    responses = response.get("responses")
    if not isinstance(problem_id, str) or not problem_id:
        raise TypeError("FrontierScience response requires a problem ID")
    if not isinstance(problem, dict):
        raise TypeError("FrontierScience response requires public problem fields")
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
        benchmark="frontierscience",
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
            "verified": validation.get("mode") == "frontierscience"
            and validation.get("status") == "passed",
            "judge_model": validation.get("judge_model"),
        },
        metadata=problem,
        provenance={
            "source_response": str(response_path.relative_to(trial.parent)),
            "dataset_revision": REVISION,
            "harness_commit": HARNESS_COMMIT,
            "content_sha256": sha256(encoded.encode()).hexdigest(),
            "response_sha256": sha256(response_path.read_bytes()).hexdigest(),
        },
    )


def sft_samples(trajectory: Trajectory, view: str) -> tuple[SftSample, ...]:
    if view not in ("full", "native"):
        raise ValueError(f"{view!r} view is unavailable for FrontierScience")
    return native_samples(trajectory)
