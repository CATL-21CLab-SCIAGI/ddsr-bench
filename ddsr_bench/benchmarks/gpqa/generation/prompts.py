from __future__ import annotations

from importlib import resources

from ddsr_bench.benchmarks.gpqa.data.schemas import ProblemSpec
from ddsr_bench.generation.client import ChatMessages

SYSTEM = "You are a very intelligent assistant, who follows instructions directly."


def render(problem: ProblemSpec) -> str:
    """Render the official zero-shot baseline prompt byte-for-byte."""
    template = (
        resources.files("configs")
        .joinpath("prompts", "gpqa", "zero-shot.txt")
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )
    values = {
        "{question}": problem.question,
        **{
            f"{{choice_{letter.lower()}}}": choice
            for letter, choice in zip("ABCD", problem.choices, strict=True)
        },
    }
    for key, value in values.items():
        template = template.replace(key, value)
    return template


def messages(problem: ProblemSpec) -> ChatMessages:
    return (
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": render(problem)},
    )
