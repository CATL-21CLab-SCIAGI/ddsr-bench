from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.critpt.prompts import system_prompt
from ddsr_bench.training.schemas import Generation, SftSample, Trajectory
from ddsr_bench.training.sft import native_samples

PROMPT_COMMIT = "17c2545c302762d2f2d644d923ea4c301605cb08"


def _messages(record: dict[str, Any]) -> tuple[dict[str, str], ...]:
    messages = record.get("messages")
    responses = record.get("responses")
    if not isinstance(messages, list) or not isinstance(responses, list):
        raise TypeError("response record must contain message and response lists")

    result = []
    response_index = 0
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("each message must be a JSON object")
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            raise TypeError(
                "each message must contain a supported role and text content"
            )
        item = {"role": role, "content": content}
        if role == "assistant":
            if response_index >= len(responses):
                raise ValueError("assistant messages and responses do not match")
            response = responses[response_index]
            reasoning = (
                response.get("reasoning") if isinstance(response, dict) else None
            )
            if isinstance(reasoning, str) and reasoning:
                item["reasoning"] = reasoning
            response_index += 1
        result.append(item)
    if response_index != len(responses):
        raise ValueError("assistant messages and responses do not match")
    return tuple(result)


def _generations(
    record: dict[str, Any], messages: tuple[dict[str, str], ...]
) -> tuple[Generation, ...]:
    assistants = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
    ]
    names = {
        "one-step": ("answer",),
        "two-step": ("derivation", "formatting"),
    }.get(record.get("strategy"))
    if names is None or len(assistants) != len(names):
        raise ValueError("response calls do not match the CritPt strategy")

    return tuple(
        Generation(
            name,
            tuple(
                {"role": message["role"], "content": message["content"]}
                for message in messages[:index]
            ),
            messages[index],
            "model",
            quality={},
        )
        for name, index in zip(names, assistants, strict=True)
    )


def _identity(problem_id: str) -> tuple[str | None, str | None, int | None]:
    sub = re.fullmatch(r"(.+)_sub_(\d+)", problem_id)
    if sub:
        return sub.group(1), "sub", int(sub.group(2))
    if problem_id.endswith("_main"):
        return problem_id.removesuffix("_main"), "main", None
    return None, None, None


def _problem(record: dict[str, Any]) -> tuple[str, str]:
    """Read public fields, including from older CritPt response records."""
    problem = record.get("problem")
    if isinstance(problem, dict):
        statement = problem.get("statement")
        template = problem.get("code_template")
        if isinstance(statement, str) and isinstance(template, str):
            return statement, template

    messages = record.get("messages")
    strategy = record.get("strategy")
    if not isinstance(messages, list) or len(messages) < 2:
        raise TypeError("response record requires public problem fields")
    statement = messages[1].get("content")
    template_source = (
        statement if strategy == "one-step" else messages[-2].get("content")
    )
    marker = "\n```python\n"
    if not isinstance(statement, str) or not isinstance(template_source, str):
        raise TypeError("response record requires public problem fields")
    prefix, separator, fenced = template_source.rpartition(marker)
    if not separator or not fenced.endswith("\n```"):
        raise ValueError("cannot recover the code template from response messages")
    if strategy == "one-step":
        statement = prefix.rstrip()
    return statement, fenced.removesuffix("\n```")


def normalize(
    trial: Path,
    response: dict[str, Any],
    result: dict[str, Any],
    validation: dict[str, Any],
) -> Trajectory:
    """Normalize one CritPt trial without exposing verifier inputs."""
    problem_id = response.get("problem_id")
    if not isinstance(problem_id, str) or not problem_id:
        raise TypeError("response record requires a problem ID")
    challenge_id, problem_type, problem_index = _identity(problem_id)
    statement, code_template = _problem(response)
    messages = _messages(response)
    kwargs = ((result.get("config") or {}).get("agent") or {}).get("kwargs") or {}
    mode, reward = validation.get("mode"), validation.get("reward")
    response_path = trial / "agent" / "response.json"
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
    return Trajectory(
        schema_version=2,
        id=result.get("trial_name") or trial.name,
        benchmark="critpt",
        problem_id=problem_id,
        generations=_generations(response, messages),
        teacher={
            "client": kwargs.get("client_name"),
            "model": response.get("model"),
            "strategy": response.get("strategy"),
            "sampling": response.get("sampling"),
        },
        quality={
            "finish_reasons": [
                item.get("finish_reason") for item in response["responses"]
            ],
            "validation": mode,
            "status": validation.get("status"),
            "reward": reward,
            "verified": mode not in ("format", "static") and reward == 1,
        },
        metadata={
            "challenge_id": challenge_id,
            "problem_type": problem_type,
            "problem_index": problem_index,
            "statement": statement,
            "code_template": code_template,
        },
        provenance={
            "source_response": str(response_path.relative_to(trial.parent)),
            "prompt_commit": PROMPT_COMMIT,
            "content_sha256": sha256(encoded).hexdigest(),
            "response_sha256": sha256(response_path.read_bytes()).hexdigest(),
        },
    )


def _answer_sample(
    trajectory: Trajectory, recorded: tuple[SftSample, ...]
) -> SftSample:
    """Recast a two-step result as one CritPt one-step training sample."""
    content = "\n\n".join(
        generation.completion["content"]
        for generation in trajectory.generations
        if generation.completion is not None
    )
    return SftSample(
        id=f"{trajectory.id}:answer",
        prompt=(
            {"role": "system", "content": system_prompt("one-step")},
            {
                "role": "user",
                "content": (
                    f"{trajectory.metadata['statement']}\n\n"
                    f"```python\n{trajectory.metadata['code_template']}\n```"
                ),
            },
        ),
        completion=({"role": "assistant", "content": content},),
        metadata=recorded[-1].metadata | {"stage": "answer", "derived": True},
    )


def sft_samples(trajectory: Trajectory, view: str) -> tuple[SftSample, ...]:
    """Apply CritPt's native, derived-answer, and stage-specific SFT views."""
    supported = ("full", "native", "derivation", "formatting", "answer")
    if view not in supported:
        raise ValueError(f"{view!r} view is unavailable for CritPt")

    strategy = trajectory.teacher.get("strategy")
    expected = {"one-step": 1, "two-step": 2}.get(strategy)
    if expected is None:
        raise ValueError(f"unknown CritPt strategy: {strategy!r}")
    if len(trajectory.generations) != expected:
        raise ValueError(
            f"{strategy} trajectory requires {expected} generations; "
            f"found {len(trajectory.generations)}"
        )

    recorded = native_samples(trajectory)
    if len(recorded) != expected:
        raise ValueError(f"{strategy} trajectory has incomplete model generations")
    if view == "native":
        return recorded

    samples = {sample.metadata["stage"]: sample for sample in recorded}
    if strategy == "two-step":
        samples["answer"] = _answer_sample(trajectory, recorded)
    if view == "full":
        return tuple(samples.values())
    if view not in samples:
        raise ValueError(f"{view!r} view is unavailable for this trajectory")
    return (samples[view],)
