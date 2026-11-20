from __future__ import annotations

import re
import string
from importlib import resources

from ddsr_bench.benchmarks.mmmupro.data.schemas import Mode, ProblemSpec
from ddsr_bench.generation.client import ChatMessages

SYSTEM = (
    "You are ChatGPT, a large language model trained by OpenAI, "
    "based on the GPT-4 architecture."
)


def _prompt(name: str) -> str:
    return (
        resources.files("configs")
        .joinpath("prompts", "mmmupro", name)
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )


def render(problem: ProblemSpec, mode: Mode) -> tuple[str, tuple[str, ...]]:
    """Render the official prompt and image ordering for one setting."""
    if problem.setting == "vision":
        return _prompt(f"{mode}-vision.txt"), problem.images
    choices = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(
            string.ascii_uppercase[: len(problem.choices)],
            problem.choices,
            strict=True,
        )
    )
    text = f"{problem.question}\n{choices}\n{_prompt(f'{mode}-standard.txt')}"
    order = [int(index) for index in re.findall(r"<image\s+(\d+)>", text)]
    text = re.sub(r"<image\s+\d+>", "<image>", text)
    try:
        images = tuple(problem.images[index - 1] for index in order)
    except IndexError as error:
        raise ValueError(f"{problem.id} references an unavailable image") from error
    if not images:
        raise ValueError(f"{problem.id} prompt references no images")
    return text, images


def messages(problem: ProblemSpec, mode: Mode) -> ChatMessages:
    text, images = render(problem, mode)
    content: list[dict[str, object]] = [{"type": "text", "text": text}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image}} for image in images
    )
    return (
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    )
