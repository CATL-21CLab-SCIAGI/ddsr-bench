from __future__ import annotations

import re
from importlib import resources

from ddsr_bench.benchmarks.hle.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.generation.client import ChatMessages


def _prompt(name: str) -> str:
    return (
        resources.files("configs")
        .joinpath("prompts", "hle", name)
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )


def system_prompt() -> str:
    """Return the exact pinned HLE response-format prompt."""
    return _prompt("system.txt")


def messages(problem: ProblemSpec, model: str) -> ChatMessages:
    """Build the upstream text-or-multimodal conversation.

    The original harness sends the format prompt as a user message only to
    legacy o1 endpoints, which did not accept a system role.
    """
    content: list[dict[str, object]] = [{"type": "text", "text": problem.question}]
    if problem.image:
        content.append({"type": "image_url", "image_url": {"url": problem.image}})
    return (
        {
            "role": "user" if "o1" in model else "system",
            "content": system_prompt(),
        },
        {"role": "user", "content": content},
    )


def judge_prompt(problem: ProblemSpec, response: str, answer: AnswerSpec) -> str:
    """Render the exact pinned HLE judge prompt."""
    values = {
        "{question}": problem.question,
        "{response}": response,
        "{correct_answer}": answer.answer,
    }
    parts = re.split(
        r"(\{question\}|\{response\}|\{correct_answer\})", _prompt("judge.txt")
    )
    return "".join(values.get(part, part) for part in parts)
