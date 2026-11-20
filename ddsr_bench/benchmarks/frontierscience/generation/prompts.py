from __future__ import annotations

import re
from importlib import resources

from ddsr_bench.benchmarks.frontierscience.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.generation.client import ChatMessages


def messages(problem: ProblemSpec) -> ChatMessages:
    """Use the complete official problem field as the single user turn."""
    return ({"role": "user", "content": problem.prompt},)


def judge_prompt(problem: ProblemSpec, response: str, answer: AnswerSpec) -> str:
    """Render the paper's exact track-specific judge prompt."""
    name = f"{problem.track}.txt"
    template = (
        resources.files("configs")
        .joinpath("prompts", "frontierscience", name)
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )
    values = {
        "{problem}": problem.prompt,
        "{answer}": response,
        "{reference_answer}": answer.reference,
        "{rubric}": answer.reference,
    }
    parts = re.split(
        r"(\{problem\}|\{answer\}|\{reference_answer\}|\{rubric\})",
        template,
    )
    return "".join(values.get(part, part) for part in parts)
