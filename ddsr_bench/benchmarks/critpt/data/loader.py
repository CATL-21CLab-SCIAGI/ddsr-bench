"""Public loaders for CritPt sources; preparation selects the source."""

from dataclasses import replace
from pathlib import Path

from ddsr_bench.benchmarks.utils import read_json

from .ai import load_problem as load_ai_problem
from .ai import load_problems as load_ai_problems
from .official import load_challenge, load_challenges, load_problems
from .schemas import AIAnswerSpec, AIProblem, AIProblemSpec
from .utils import decode_instruction


def load_ai_task(task_dir: str | Path) -> AIProblem:
    """Restore normalized public/private data; never execute reference code."""
    task = Path(task_dir)
    spec = decode_instruction((task / "instruction.md").read_text(encoding="utf-8"))
    if not isinstance(spec, AIProblemSpec):
        raise TypeError("AI judging requires a critpt-ai task")
    reference = read_json(task / "tests/reference.json", allow_nan=False)
    source = reference.pop("source_path", None)
    if source is not None:
        spec = replace(spec, source_path=Path(source))
    testcases = reference.pop("testcases", None)
    answer = AIAnswerSpec(
        **reference, testcases=None if testcases is None else tuple(testcases)
    )
    return AIProblem(spec, answer)


__all__ = [
    "load_ai_problem",
    "load_ai_problems",
    "load_ai_task",
    "load_challenge",
    "load_challenges",
    "load_problems",
]
