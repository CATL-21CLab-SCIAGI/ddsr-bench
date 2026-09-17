from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from ddsr_bench.benchmarks.critpt.data.schemas import Challenge, ProblemSpec
from ddsr_bench.benchmarks.critpt.generation.prompts import (
    PromptStyle,
    parse_prompt,
    system_prompt,
)
from ddsr_bench.generation.client import ChatClient, ChatResponse, Sampling


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class Conversation:
    content: str
    messages: tuple[Message, ...]


Completion = Callable[[tuple[Message, ...]], Awaitable[str]]


def _user_message(problem: ProblemSpec, style: PromptStyle) -> Message:
    content = problem.statement
    if style == "one-step":
        content = f"{content}\n\n```python\n{problem.code_template}\n```"
    return Message("user", content)


def _initial_messages(problem: ProblemSpec, style: PromptStyle) -> list[Message]:
    return [Message("system", system_prompt(style)), _user_message(problem, style)]


async def _converse(
    problem: ProblemSpec,
    style: PromptStyle,
    complete: Completion,
    history: tuple[Message, ...] = (),
) -> tuple[Conversation, tuple[Message, ...]]:
    messages = list(history) if history else _initial_messages(problem, style)
    if history:
        messages.append(_user_message(problem, style))
    response = await complete(tuple(messages))
    messages.append(Message("assistant", response))
    next_history = tuple(messages)

    if style == "two-step":
        messages.append(Message("user", parse_prompt(problem.code_template)))
        response = await complete(tuple(messages))
        messages.append(Message("assistant", response))

    return Conversation(content=response, messages=tuple(messages)), next_history


async def converse(
    problem: ProblemSpec, style: PromptStyle, complete: Completion
) -> Conversation:
    """Apply CritPt's one-step or two-step conversation policy."""
    result, _ = await _converse(problem, style, complete)
    return result


async def converse_challenge(
    challenge: Challenge,
    style: PromptStyle,
    complete: Completion,
    *,
    use_golden: bool = False,
) -> tuple[Conversation, ...]:
    """Generate the main independently, then the challenge's ordered subproblems."""
    main = await converse(challenge.main.spec, style, complete)
    subs = challenge.problems[1:]
    if use_golden:
        results = await asyncio.gather(
            *(converse(problem.spec, style, complete) for problem in subs)
        )
        return (main, *results)

    history: tuple[Message, ...] = ()
    results = []
    for problem in subs:
        result, history = await _converse(problem.spec, style, complete, history)
        results.append(result)
    return (main, *results)


def _hash(content: str) -> str:
    return sha256(content.encode()).hexdigest()


async def generate(
    client: ChatClient,
    problem: ProblemSpec,
    style: PromptStyle,
    sampling: Sampling,
    *,
    formatting_max_tokens: int | None = None,
    checkpoint_dir: Path | None = None,
    require_complete_stages: bool = False,
    resume: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Run one conversation and return its public audit record."""
    responses: list[ChatResponse] = []

    def record(messages, status):
        return {
            "problem_id": problem.id,
            "problem": {
                "statement": problem.statement,
                "code_template": problem.code_template,
            },
            "strategy": style,
            "model": sampling.model,
            "seed": sampling.seed,
            "sampling": asdict(sampling),
            "formatting_max_tokens": formatting_max_tokens,
            "require_complete_stages": require_complete_stages,
            "status": status,
            "messages": [asdict(m) | {"sha256": _hash(m.content)} for m in messages],
            "responses": [asdict(r) | {"sha256": _hash(r.content)} for r in responses],
        }

    def save(path, data):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)

    async def complete(messages: tuple[Message, ...]) -> str:
        stage = len(responses) + 1
        checkpoint = checkpoint_dir / f"stage-{stage}.json" if checkpoint_dir else None
        if resume and checkpoint is not None and checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            expected = record(messages, "in_progress")
            for key in (
                "problem_id",
                "problem",
                "strategy",
                "sampling",
                "require_complete_stages",
            ):
                if saved.get(key) != expected[key]:
                    raise ValueError(f"cannot resume stage {stage}: {key} changed")
            # Stage one's request is independent of the later formatting cap.
            # A saved stage two must still match its original output budget.
            if (
                stage > 1
                and saved.get("formatting_max_tokens") != formatting_max_tokens
            ):
                raise ValueError(
                    f"cannot resume stage {stage}: formatting_max_tokens changed"
                )
            previous_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in saved["messages"][:-1]
            ]
            if (
                previous_messages != [asdict(m) for m in messages]
                or len(saved["responses"]) != stage
            ):
                raise ValueError(f"cannot resume stage {stage}: conversation changed")
            value = saved["responses"][-1]
            response = ChatResponse(
                **{
                    f.name: value[f.name]
                    for f in fields(ChatResponse)
                    if f.name in value
                }
            )
            responses.append(response)
            if require_complete_stages and (
                response.finish_reason not in (None, "stop")
                or not response.content.strip()
            ):
                raise ValueError(f"saved stage {stage} did not complete normally")
            return response.content
        kwargs = {}
        if sampling.seed is not None:
            kwargs["seed"] = sampling.seed
        if style == "two-step" and stage == 2 and formatting_max_tokens is not None:
            kwargs["max_tokens"] = formatting_max_tokens
        if checkpoint_dir is not None:
            journal = checkpoint_dir / f"stage-{stage}.stream.jsonl"
            if resume and journal.exists():
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                journal.rename(
                    checkpoint_dir / f"stage-{stage}.interrupted-{stamp}.stream.jsonl"
                )
            kwargs["stream_path"] = journal
        response = await client.chat(
            tuple(asdict(message) for message in messages), **kwargs
        )
        responses.append(response)
        if checkpoint_dir is not None:
            snapshot = record(
                (*messages, Message("assistant", response.content)), "in_progress"
            )
            save(checkpoint_dir / f"stage-{stage}.json", snapshot)
            save(checkpoint_dir / "response.json", snapshot)
        if require_complete_stages and response.finish_reason not in (None, "stop"):
            raise ValueError(
                f"stage {stage} did not finish normally: {response.finish_reason}"
            )
        if require_complete_stages and not response.content.strip():
            raise ValueError(f"stage {stage} returned no final answer content")
        return response.content

    result = await converse(problem, style, complete)
    final = record(result.messages, "completed")
    if checkpoint_dir is not None:
        save(checkpoint_dir / "response.json", final)
    return result.content, final
