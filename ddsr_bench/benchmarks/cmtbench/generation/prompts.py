from functools import cache
from importlib.resources import files

from ddsr_bench.benchmarks.cmtbench.data.schemas import ProblemSpec
from ddsr_bench.generation.client import ChatMessages


@cache
def system_prompt() -> str:
    """Load the system instruction used by the pinned CMT-Benchmark client."""
    return (
        files("configs")
        .joinpath("prompts", "cmtbench", "default.txt")
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )


def messages(problem: ProblemSpec) -> ChatMessages:
    """Preserve the official system message and raw dataset prompt."""
    return (
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": problem.prompt},
    )
