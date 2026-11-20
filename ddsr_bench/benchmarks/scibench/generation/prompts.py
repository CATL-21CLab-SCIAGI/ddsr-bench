from __future__ import annotations

import re
from importlib import resources

from ddsr_bench.benchmarks.scibench.data.schemas import Example, ProblemSpec, Strategy
from ddsr_bench.generation.client import ChatMessages


def system_prompt() -> str:
    """Return the official system instruction, including boundary newlines."""
    return (
        resources.files("configs")
        .joinpath("prompts", "scibench", "system.txt")
        .read_text(encoding="utf-8")
    )


def display_unit(unit: str) -> str:
    pattern = re.compile(r"[$]? *10\^[{]? *-?[0-9]+ *[}]? *[$]?")
    return re.split(pattern, unit)[-1] if pattern.search(unit) else unit


def _problem_text(problem: ProblemSpec) -> str:
    return f"{problem.question} The unit of the answer is {display_unit(problem.unit)}."


def _content(text: str, image: str | None) -> str | list[dict[str, object]]:
    if image is None:
        return text
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image}},
    ]


def zero_messages(problem: ProblemSpec, derivation: str | None = None) -> ChatMessages:
    text = _problem_text(problem)
    if derivation is None:
        user = f"Q: {text}\nA: Let's think step by step."
    else:
        user = (
            f"Q: {text}\nA: Let's think step by step."
            f"{derivation}Therefore, the answer is"
        )
    return (
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": _content(user, problem.image)},
    )


def direct_messages(problem: ProblemSpec) -> ChatMessages:
    user = f"Q: {_problem_text(problem)}\nA: The answer is"
    return (
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": _content(user, problem.image)},
    )


def _example_messages(example: Example, index: int, cot: bool) -> ChatMessages:
    question = f"Problem {index}.   {example.question} The unit of the answer is {example.unit}.\n"
    answer = f"The answer is \\boxed{{{example.answer}}} {example.unit}"
    if cot:
        answer = f"Explanation for Problem {index}:   {example.solution}\n{answer}"
    return (
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    )


def few_messages(problem: ProblemSpec, cot: bool) -> ChatMessages:
    if problem.split != "text" or len(problem.examples) < 3:
        raise ValueError("SciBench few-shot strategies require three text examples")
    messages: list[dict[str, object]] = [{"role": "system", "content": system_prompt()}]
    for index, example in enumerate(problem.examples[:3], start=1):
        messages.extend(_example_messages(example, index, cot))
    messages.append(
        {
            "role": "user",
            "content": f"Problem 4.   {_problem_text(problem)}\n",
        }
    )
    return tuple(messages)


def initial_messages(problem: ProblemSpec, strategy: Strategy) -> ChatMessages:
    if strategy == "zero":
        return direct_messages(problem)
    if strategy == "zero-cot":
        return zero_messages(problem)
    if strategy == "few":
        return few_messages(problem, False)
    if strategy == "few-cot":
        return few_messages(problem, True)
    raise ValueError(f"unknown SciBench strategy {strategy!r}")
