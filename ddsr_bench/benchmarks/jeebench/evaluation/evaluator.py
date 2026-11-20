from __future__ import annotations

import re
from typing import Any

from ddsr_bench.benchmarks.jeebench.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.benchmarks.utils import benchmark_config

_BOXED = re.compile(r"\\boxed\s*\{([^{}]+)\}", re.IGNORECASE)
_FINAL = re.compile(
    r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)?\s*([^\n]+)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_FRACTION = re.compile(r"\\frac\s*\{([-+]?\d+(?:\.\d*)?)\}\s*\{([-+]?\d+(?:\.\d*)?)\}")
_SCIENTIFIC = re.compile(
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:\\times|×|\*)\s*10\s*\^\s*\{?([-+]?\d+)\}?"
)
_TOLERANCE = float(benchmark_config("jeebench")["numeric_tolerance"])


def _tail(response: str) -> str:
    boxed = _BOXED.findall(response)
    if boxed:
        return boxed[-1]
    finals = _FINAL.findall(response)
    return finals[-1] if finals else response.rsplit("\n", maxsplit=1)[-1]


def parse_answer(response: str, response_type: str) -> str | None:
    """Extract the final value; upstream publishes extracts but not its extractor."""
    tail = _tail(response)
    if response_type in ("MCQ", "MCQ(multiple)"):
        groups = re.findall(r"(?<![A-Z])([A-D]{1,4})(?![A-Z])", tail.upper())
        letters = "".join(dict.fromkeys("".join(groups)))
        return "".join(sorted(letters)) or None
    scientific = _SCIENTIFIC.findall(tail)
    if scientific:
        coefficient, exponent = scientific[-1]
        return str(float(coefficient) * 10 ** int(exponent))
    fractions = _FRACTION.findall(tail)
    if fractions:
        numerator, denominator = fractions[-1]
        if float(denominator) != 0:
            return str(float(numerator) / float(denominator))
    numbers = _NUMBER.findall(tail.replace(",", ""))
    return numbers[-1] if numbers else None


def score(expected: str, selected: str | None, response_type: str) -> float:
    """Apply the official JEEBench scoring rule."""
    if selected is None:
        return 0.0
    if response_type == "MCQ(multiple)":
        gold, response = set(expected) & set("ABCD"), set(selected) & set("ABCD")
        if response == gold:
            return 1.0
        return 0.25 * len(response) if response and response <= gold else 0.0
    if response_type == "MCQ":
        return float((set(expected) & set("ABCD")) == (set(selected) & set("ABCD")))
    try:
        return float(abs(float(expected) - float(selected)) <= _TOLERANCE)
    except ValueError:
        return 0.0


def evaluate(response: str, problem: ProblemSpec, answer: AnswerSpec) -> dict[str, Any]:
    selected = parse_answer(response, problem.response_type)
    reward = score(answer.value, selected, problem.response_type)
    status = "error" if selected is None else "passed" if reward == 1 else "different"
    result: dict[str, Any] = {
        "reward": reward,
        "mode": "jeebench",
        "status": status,
        "paper": problem.paper,
        "subject": problem.subject,
        "response_type": problem.response_type,
        "selected": selected,
    }
    if selected is None:
        result["error"] = "AnswerFormatError"
    return result
