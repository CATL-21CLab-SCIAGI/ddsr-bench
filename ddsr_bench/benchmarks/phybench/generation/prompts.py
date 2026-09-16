from functools import cache
from importlib.resources import files

from ddsr_bench.benchmarks.phybench.data.schemas import ProblemSpec
from ddsr_bench.generation.client import ChatMessages

_CONTENT = "{content}"


@cache
def prompt_template() -> str:
    """Load the prompt published in PHYBench's arXiv v2 appendix."""
    template = (
        files("configs")
        .joinpath("prompts", "phybench", "default.txt")
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )
    if template.count(_CONTENT) != 1:
        raise ValueError("PHYBench prompt requires one content field")
    return template


def user_prompt(problem: ProblemSpec) -> str:
    return prompt_template().replace(_CONTENT, problem.content)


def messages(problem: ProblemSpec) -> ChatMessages:
    return ({"role": "user", "content": user_prompt(problem)},)
