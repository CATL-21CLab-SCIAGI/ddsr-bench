from functools import cache
from importlib.resources import files

from ddsr_bench.benchmarks.prismphysics.data.schemas import ProblemSpec
from ddsr_bench.generation.client import ChatMessages


@cache
def _asset(name: str) -> str:
    return (
        files("configs")
        .joinpath("prompts", "prismphysics", name)
        .read_text(encoding="utf-8")
    )


def _context(problem: ProblemSpec) -> str:
    context = problem.context
    for subquestion in problem.subquestions:
        context += f"\n\n{subquestion.letter}) {subquestion.prompt}"
    return context


def prompt(problem: ProblemSpec) -> str:
    """Render the pinned text-only prompt byte-for-byte.

    Upstream's non-raw example emits a form-feed before ``rac``. Its renderer
    also looks for singular instruction keys absent from the released data, so
    the plural final-answer instructions are intentionally not appended.
    """
    return (
        _asset("template.txt")
        .removesuffix("\n")
        .replace("__FORM_FEED__", "\f")
        .replace("__FORMAT__", _asset("default.txt"))
        .replace("__CONTEXT__", _context(problem))
    )


def messages(problem: ProblemSpec) -> ChatMessages:
    """Use the official single user message without a system message."""
    return ({"role": "user", "content": prompt(problem)},)
