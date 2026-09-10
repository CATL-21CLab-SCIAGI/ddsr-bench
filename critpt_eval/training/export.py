from __future__ import annotations

import json
import re
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from critpt_eval.generation.prompts import system_prompt
from critpt_eval.schemas import SftSample, Trajectory

PROMPT_COMMIT = "17c2545c302762d2f2d644d923ea4c301605cb08"
SftView = Literal["full", "derivation", "formatting", "answer"]


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON object from {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


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
            # Keep provider-specific internal reasoning in the canonical record.
            # Generic conversational SFT below supervises visible content only.
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


def _identity(problem_id: str) -> tuple[str | None, str | None, int | None]:
    sub = re.fullmatch(r"(.+)_sub_(\d+)", problem_id)
    if sub:
        return sub.group(1), "sub", int(sub.group(2))
    if problem_id.endswith("_main"):
        return problem_id.removesuffix("_main"), "main", None
    return None, None, None


def _problem(record: dict[str, Any]) -> tuple[str, str]:
    """Read public problem fields, including from older response records."""
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


def load_trajectory(trial_dir: str | Path) -> Trajectory:
    """Load one canonical trajectory from a static or Harbor trial directory."""
    trial = Path(trial_dir)
    response_path = trial / "agent" / "response.json"
    response_record = _json(response_path)
    trial_record = _json(trial / "result.json")
    validation_path = trial / "validation" / "result.json"
    if not validation_path.exists():
        validation_path = trial / "verifier" / "result.json"
    validation = _json(validation_path)

    problem_id = response_record.get("problem_id")
    if not isinstance(problem_id, str) or not problem_id:
        raise TypeError("response record requires a problem ID")
    challenge_id, problem_type, problem_index = _identity(problem_id)
    statement, code_template = _problem(response_record)
    messages = _messages(response_record)
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
    kwargs = ((trial_record.get("config") or {}).get("agent") or {}).get("kwargs") or {}
    mode, reward = validation.get("mode"), validation.get("reward")
    return Trajectory(
        schema_version=1,
        id=trial_record.get("trial_name") or trial.name,
        challenge_id=challenge_id,
        problem_id=problem_id,
        problem_type=problem_type,
        problem_index=problem_index,
        statement=statement,
        code_template=code_template,
        messages=messages,
        teacher={
            "client": kwargs.get("client_name"),
            "model": response_record.get("model"),
            "strategy": response_record.get("strategy"),
            "sampling": response_record.get("sampling"),
        },
        quality={
            "finish_reasons": [
                item.get("finish_reason") for item in response_record["responses"]
            ],
            "validation": mode,
            "status": validation.get("status"),
            "reward": reward,
            "verified": mode not in ("format", "static") and reward == 1,
        },
        provenance={
            "source_response": str(response_path.relative_to(trial.parent)),
            "prompt_commit": PROMPT_COMMIT,
            "content_sha256": sha256(encoded).hexdigest(),
            "response_sha256": sha256(response_path.read_bytes()).hexdigest(),
        },
    )


def export_trajectories(job_dir: str | Path, output: str | Path) -> Path:
    """Write deterministic canonical trajectories for one completed job."""
    job = Path(job_dir)
    trials = sorted(path.parent.parent for path in job.glob("*/agent/response.json"))
    if not trials:
        raise ValueError(f"no response records found in {job}")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "trajectories.jsonl"
    with target.open("w", encoding="utf-8") as file:
        for trial in trials:
            json.dump(asdict(load_trajectory(trial)), file, ensure_ascii=False)
            file.write("\n")
    return target


def sft_samples(
    trajectory: Trajectory, view: SftView = "full"
) -> tuple[SftSample, ...]:
    """Create explicitly supervised assistant stages.

    ``full`` exports every available projection. ``derivation`` selects the
    visible content from the first CritPt call, not a provider's internal
    reasoning field. ``formatting`` preserves the native second step.
    ``answer`` merges the visible derivation and code under a one-step prompt.
    """
    if view not in ("full", "derivation", "formatting", "answer"):
        raise ValueError(f"unknown SFT view: {view!r}")
    assistants = [
        index
        for index, message in enumerate(trajectory.messages)
        if message["role"] == "assistant"
    ]
    strategy = trajectory.teacher.get("strategy")
    expected = {"one-step": 1, "two-step": 2}.get(strategy)
    if expected is None:
        raise ValueError(f"unknown CritPt strategy: {strategy!r}")
    if len(assistants) != expected:
        raise ValueError(
            f"{strategy} trajectory requires {expected} assistant turns; "
            f"found {len(assistants)}"
        )

    stages = [("answer", (assistants[0],))]
    if strategy == "two-step":
        stages = [
            ("derivation", (assistants[0],)),
            ("formatting", (assistants[1],)),
            ("answer", tuple(assistants)),
        ]
    if view != "full":
        stages = [item for item in stages if item[0] == view]
        if not stages:
            raise ValueError(f"{view!r} view is unavailable for this trajectory")

    samples = []
    for stage, response_indices in stages:
        if stage == "answer":
            prompt = (
                {"role": "system", "content": system_prompt("one-step")},
                {
                    "role": "user",
                    "content": (
                        f"{trajectory.statement}\n\n"
                        f"```python\n{trajectory.code_template}\n```"
                    ),
                },
            )
        else:
            index = response_indices[0]
            prompt = tuple(
                {"role": message["role"], "content": message["content"]}
                for message in trajectory.messages[:index]
            )
        # Completions always come from a model response's visible content.
        content = "\n\n".join(
            trajectory.messages[index]["content"] for index in response_indices
        )
        completion = ({"role": "assistant", "content": content},)
        samples.append(
            SftSample(
                id=f"{trajectory.id}:{stage}",
                prompt=prompt,
                completion=completion,
                metadata={
                    "trajectory_id": trajectory.id,
                    "problem_id": trajectory.problem_id,
                    "stage": stage,
                    "derived": strategy == "two-step" and stage == "answer",
                    "teacher": trajectory.teacher,
                    "quality": trajectory.quality,
                },
            )
        )
    return tuple(samples)


def export_sft(
    trajectories: str | Path,
    output: str | Path,
    view: SftView = "full",
) -> Path:
    """Stream canonical trajectories into conversational SFT JSONL."""
    source = Path(trajectories)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "sft.jsonl"
    with (
        source.open(encoding="utf-8") as input_file,
        target.open("w", encoding="utf-8") as output_file,
    ):
        for number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                data["messages"] = tuple(data["messages"])
                trajectory = Trajectory(**data)
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid trajectory at line {number}") from error
            for sample in sft_samples(trajectory, view):
                json.dump(asdict(sample), output_file, ensure_ascii=False)
                output_file.write("\n")
    return target
