from __future__ import annotations

import re
from typing import Any

from ddsr_bench.benchmarks.mmlu.data.schemas import AnswerSpec, ProblemSpec

_EXPLICIT = re.compile(r"(?:final\s+)?answer\s*(?:is|:)\s*\(?([A-D])\)?", re.IGNORECASE)
_LEADING = re.compile(r"^\s*\(?([A-D])\)?(?:[\s.,]|$)", re.IGNORECASE)


def parse_answer(response: str) -> str | None:
    """Parse one generated choice without repairing malformed output."""
    matches = _EXPLICIT.findall(response)
    if matches:
        return matches[-1].upper()
    match = _LEADING.search(response)
    return match.group(1).upper() if match else None


def evaluate(response: str, problem: ProblemSpec, answer: AnswerSpec) -> dict[str, Any]:
    selected = parse_answer(response)
    if selected is None:
        return {
            "reward": 0.0,
            "mode": "mmlu-generative",
            "status": "error",
            "error": "AnswerFormatError",
            "subject": problem.subject,
            "category": problem.category,
            "selected": None,
        }
    passed = selected == answer.letter
    return {
        "reward": float(passed),
        "mode": "mmlu-generative",
        "status": "passed" if passed else "different",
        "subject": problem.subject,
        "category": problem.category,
        "selected": selected,
    }
