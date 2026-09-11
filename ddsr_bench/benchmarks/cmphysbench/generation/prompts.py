from functools import cache
from importlib.resources import files

from ddsr_bench.benchmarks.cmphysbench.data.schemas import ProblemSpec
from ddsr_bench.generation.client import ChatMessages


@cache
def system_prompt() -> str:
    """Load the pinned CMPhysBench system prompt."""
    return (
        files("configs")
        .joinpath("prompts", "cmphysbench", "default.txt")
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )


def user_prompt(problem: ProblemSpec) -> str:
    """Reproduce CMPhysBench's input concatenation exactly."""
    question = problem.context + problem.question
    symbols = "Here are the relevant symbols:\n" + problem.symbols
    return question + "\n" + symbols


def messages(problem: ProblemSpec) -> ChatMessages:
    return (
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(problem)},
    )
