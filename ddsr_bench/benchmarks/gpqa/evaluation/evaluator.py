from __future__ import annotations

import re
from typing import Any

from ddsr_bench.benchmarks.gpqa.data.schemas import AnswerSpec, ProblemSpec

_PATTERNS = (
    re.compile(r"answer is \((.)\)", re.IGNORECASE),
    re.compile(r"answer: \((.)\)", re.IGNORECASE),
    re.compile(r"answer \((.)\)", re.IGNORECASE),
    re.compile(r"\((.)\)"),
)


def parse_answer(response: str) -> str | None:
    """Apply the official sampled-answer parser without repair prompts."""
    for pattern in _PATTERNS:
        match = pattern.search(response)
        if match is not None and match.group(1).upper() in "ABCD":
            return match.group(1).upper()
    return None


def evaluate(response: str, problem: ProblemSpec, answer: AnswerSpec) -> dict[str, Any]:
    selected = parse_answer(response)
    if selected is None:
        return {
            "reward": 0.0,
            "mode": "gpqa",
            "status": "error",
            "error": "AnswerFormatError",
            "cohort": problem.cohort,
            "domain": problem.domain,
            "subdomain": problem.subdomain,
            "selected": None,
        }
    passed = selected == answer.letter
    return {
        "reward": float(passed),
        "mode": "gpqa",
        "status": "passed" if passed else "different",
        "cohort": problem.cohort,
        "domain": problem.domain,
        "subdomain": problem.subdomain,
        "selected": selected,
    }
