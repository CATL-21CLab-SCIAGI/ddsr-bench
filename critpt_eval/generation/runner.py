from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal

from critpt_eval.generation.client import ChatClient, ChatResponse, Sampling
from critpt_eval.generation.prompts import PromptStyle, parse_prompt, system_prompt
from critpt_eval.schemas import Challenge, ProblemSpec


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
) -> tuple[str, dict[str, Any]]:
    """Run one conversation and return its public audit record."""
    responses: list[ChatResponse] = []

    async def complete(messages: tuple[Message, ...]) -> str:
        response = await client.chat(tuple(asdict(message) for message in messages))
        responses.append(response)
        return response.content

    result = await converse(problem, style, complete)
    messages = [
        asdict(message) | {"sha256": _hash(message.content)}
        for message in result.messages
    ]
    calls = [
        asdict(response) | {"sha256": _hash(response.content)} for response in responses
    ]
    return result.content, {
        "problem_id": problem.id,
        "problem": {
            "statement": problem.statement,
            "code_template": problem.code_template,
        },
        "strategy": style,
        "model": sampling.model,
        "seed": sampling.seed,
        "sampling": asdict(sampling),
        "messages": messages,
        "responses": calls,
    }
