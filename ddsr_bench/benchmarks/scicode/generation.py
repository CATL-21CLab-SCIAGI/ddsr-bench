from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

from ddsr_bench.benchmarks.scicode.schemas import SciCodeProblem
from ddsr_bench.generation.client import ChatClient, ChatResponse

from .loader import load_fixed


@dataclass(frozen=True, slots=True)
class StepGeneration:
    id: str
    prompt: str | None
    response: ChatResponse | None
    code: str
    source: Literal["model", "fixed"]


def _template(with_background: bool) -> str:
    name = "with_background.txt" if with_background else "without_background.txt"
    template = (
        files("configs")
        .joinpath("prompts", "scicode", name)
        .read_text(encoding="utf-8")
    )
    # The upstream with-background template has no final newline.
    return template.removesuffix("\n") if with_background else template


def extract_code(response: str) -> str:
    """Follow SciCode's first-fenced-block and import-removal behavior."""
    if "```" in response:
        marker = "```python" if "```python" in response else "```"
        code = response.split(marker, 1)[1].split("```", 1)[0]
    else:
        code = response
    return re.sub(
        r"^\s*(import .*|from .*\s+import\s+.*)",
        "",
        code,
        flags=re.MULTILINE,
    )


def render_prompt(
    problem: SciCodeProblem,
    step_index: int,
    previous_code: tuple[str, ...] = (),
    *,
    with_background: bool = False,
) -> str:
    """Render one step using SciCode's official sequential prompt."""
    if step_index not in range(len(problem.steps)):
        raise IndexError("SciCode step index is out of range")
    if len(previous_code) != step_index:
        raise ValueError("previous code must contain every preceding step")

    earlier = []
    for step, code in zip(problem.steps[:step_index], previous_code, strict=True):
        description = step.statement
        if with_background:
            description += "\n" + step.background
        earlier.extend((description, code, "------"))

    step = problem.steps[step_index]
    description = step.statement
    if with_background:
        description += "\n" + step.background
    next_step = f"{description}\n\n{step.function}\n\n{step.return_line}"
    return _template(with_background).format(
        problem_steps_str="\n\n".join(earlier[:-1]),
        next_step_str=next_step,
        dependencies=problem.dependencies,
    )


async def generate(
    problem: SciCodeProblem,
    client: ChatClient,
    *,
    with_background: bool = False,
) -> tuple[StepGeneration, ...]:
    """Generate every SciCode step in order with accumulated code context."""
    results = []
    code = []
    for index, step in enumerate(problem.steps):
        fixed = load_fixed(step.id)
        if fixed is not None:
            results.append(StepGeneration(step.id, None, None, fixed, "fixed"))
            code.append(fixed)
            continue
        prompt = render_prompt(
            problem, index, tuple(code), with_background=with_background
        )
        response = await client.chat(({"role": "user", "content": prompt},))
        generated = extract_code(response.content)
        results.append(StepGeneration(step.id, prompt, response, generated, "model"))
        code.append(generated)
    return tuple(results)
