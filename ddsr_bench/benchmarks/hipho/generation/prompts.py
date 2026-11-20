from __future__ import annotations

from importlib import resources
from typing import Any

from ddsr_bench.benchmarks.hipho.data.schemas import ProblemSpec
from ddsr_bench.generation.client import ChatMessages


def _template(language: str) -> str:
    return (
        resources.files("configs")
        .joinpath("prompts", "hipho", f"system-{language}.txt")
        .read_text(encoding="utf-8")
    )


def render(problem: ProblemSpec) -> str:
    """Render VLMEvalKit's exact maintained HiPhO inference prompt."""
    return (
        _template(problem.language)
        .replace("{context}", problem.context)
        .replace("{problem}", problem.question)
        .replace("{information}", problem.information)
    )


def messages(problem: ProblemSpec) -> ChatMessages:
    """Build the official image-first, single-user-message request."""
    prompt = render(problem)
    if not problem.images:
        return ({"role": "user", "content": prompt},)
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image}} for image in problem.images
    ]
    content.append({"type": "text", "text": prompt})
    return ({"role": "user", "content": content},)
