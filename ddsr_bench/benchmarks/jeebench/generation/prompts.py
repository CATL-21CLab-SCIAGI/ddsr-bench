from __future__ import annotations

from importlib import resources
from typing import Literal

from ddsr_bench.benchmarks.jeebench.data.schemas import ProblemSpec
from ddsr_bench.generation.client import ChatMessages

PromptMode = Literal["normal", "cot", "one-shot"]
SYSTEM = ""


def _prefix(response_type: str) -> str:
    filename = response_type.casefold().replace("(", "-").replace(")", "") + ".txt"
    return (
        resources.files("configs")
        .joinpath("prompts", "jeebench", filename)
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )


def render(
    problem: ProblemSpec,
    mode: PromptMode = "cot",
    example: dict[str, str] | None = None,
) -> str:
    """Render the official normal, CoT, or one-shot prompt."""
    prefix = _prefix(problem.response_type)
    if mode == "normal":
        return f"{prefix}\n\nProblem: {problem.question}"
    if mode == "cot":
        return f"{prefix}\n\nProblem: {problem.question}\nSolution: Let's think step by step."
    if mode != "one-shot":
        raise ValueError(f"unknown JEEBench prompt mode {mode!r}")
    if example is None or not all(
        isinstance(example.get(key), str) for key in ("problem", "solution")
    ):
        raise ValueError("one-shot mode requires an official example")
    return (
        f"{prefix}\n\nProblem: {example['problem']}\nSolution: {example['solution']}"
        f"\n\nProblem: {problem.question}\nSolution:"
    )


def messages(
    problem: ProblemSpec,
    mode: PromptMode = "cot",
    example: dict[str, str] | None = None,
) -> ChatMessages:
    return (
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": render(problem, mode, example)},
    )
