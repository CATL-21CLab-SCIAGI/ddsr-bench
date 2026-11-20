from __future__ import annotations

from importlib import resources

from ddsr_bench.benchmarks.mmlu.data.schemas import Example, ProblemSpec
from ddsr_bench.generation.client import ChatMessages


def _subject(subject: str) -> str:
    # Upstream deliberately returns a leading space before the first word.
    return "".join(f" {part}" for part in subject.split("_"))


def _example(example: Example, include_answer: bool) -> str:
    prompt = example.question
    for letter, choice in zip("ABCD", example.choices, strict=True):
        prompt += f"\n{letter}. {choice}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {example.answer}\n\n"
    return prompt


def render(problem: ProblemSpec) -> str:
    """Render the original five-shot completion prompt byte-for-byte."""
    header = (
        resources.files("configs")
        .joinpath("prompts", "mmlu", "header.txt")
        .read_text(encoding="utf-8")
        .replace("{subject}", _subject(problem.subject))
        + "\n"
    )
    demonstrations = "".join(_example(example, True) for example in problem.examples)
    target = _example(Example(problem.question, problem.choices, "A"), False)
    return header + demonstrations + target


def messages(problem: ProblemSpec) -> ChatMessages:
    return ({"role": "user", "content": render(problem)},)
