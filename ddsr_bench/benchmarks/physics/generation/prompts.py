from __future__ import annotations

import re
from importlib import resources

from ddsr_bench.benchmarks.physics.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.benchmarks.utils import benchmark_config
from ddsr_bench.generation.client import ChatMessages

_NUMERIC_TOLERANCE = float(benchmark_config("physics")["numeric_relative_tolerance"])
if _NUMERIC_TOLERANCE != 0.01:
    raise ValueError("the pinned PHYSICS judge prompt requires 1% numeric tolerance")


def _prompt(name: str) -> str:
    return (
        resources.files("configs")
        .joinpath("prompts", "physics", name)
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )


def system_prompt() -> str:
    """Return the exact pinned PHYSICS-Verified response-format prompt."""
    return _prompt("system.txt")


def messages(problem: ProblemSpec) -> ChatMessages:
    """Build the exact upstream text-or-multimodal conversation."""
    content: list[dict[str, object]] = [{"type": "text", "text": problem.question}]
    content.extend(
        {"type": "image_url", "image_url": {"url": figure.url}}
        for figure in problem.figures
    )
    return (
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": content},
    )


def judge_prompt(problem: ProblemSpec, response: str, answer: AnswerSpec) -> str:
    """Render the exact pinned per-problem judge prompt."""
    references = "\n".join(
        f"{index}. label {label}: {value}"
        for index, (label, value) in enumerate(
            zip(answer.labels, answer.final_answers, strict=True)
        )
    )
    values = {
        "{question}": problem.question,
        "{response}": response,
        "{references}": references,
    }
    parts = re.split(
        r"(\{question\}|\{response\}|\{references\})", _prompt("judge.txt")
    )
    return "".join(values.get(part, part) for part in parts)
