from __future__ import annotations

import math
import re
from typing import Any

from ddsr_bench.benchmarks.scibench.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.benchmarks.utils import benchmark_config

_REL_TOL = float(benchmark_config("scibench")["relative_tolerance"])


def _boxed(response: str) -> str | None:
    index = response.rfind("oxed")
    if index < 0:
        index = response.rfind("\\fbox")
    if index < 0:
        return None
    depth = 0
    end = None
    for position in range(index, len(response)):
        if response[position] == "{":
            depth += 1
        elif response[position] == "}":
            depth -= 1
            if depth == 0:
                end = position
                break
    if end is None:
        return None
    boxed = response[index : end + 1]
    prefix = "oxed{"
    if not boxed.startswith(prefix) or not boxed.endswith("}"):
        return None
    answer = boxed[len(prefix) : -1]
    return answer.split("=")[-1].lstrip() if "=" in answer else answer


def _notation(value: str) -> str | tuple[str, str]:
    try:
        if "\times" in value:
            return tuple(value.split("\times", maxsplit=1))
        if "\\times" in value:
            return tuple(value.split("\\times", maxsplit=1))
        if "*" in value:
            return tuple(value.split("*", maxsplit=1))
    except Exception:  # noqa: BLE001 - preserve upstream fallback
        return ("", "")
    return value


def _scale(value: str | tuple[str, str]) -> str:
    try:
        coefficient, exponent = list(value)
        match = re.findall(r"10\^[{]? *-?[0-9]+ *[}]?", exponent)[0]
        power = match[match.find("^") + 1 :].strip("{}")
        return str(float(coefficient.strip()) * 10 ** float(power))
    except Exception:  # noqa: BLE001 - preserve upstream fallback
        return value if isinstance(value, str) else str(value)


def _scientific_unit(unit: str) -> bool:
    return bool(re.search(r"[$]? *10\^[{]? *-?[0-9]+ *[}]? *[$]?", unit))


def _equivalent(candidate: str | None, expected: str) -> bool:
    if candidate is None:
        return False
    candidate = candidate.replace(",", "")
    try:
        target = float(expected.strip())
        first = math.isclose(float(candidate.strip()), target, rel_tol=_REL_TOL)
    except Exception:  # noqa: BLE001 - official parser treats it as a miss
        first = False
    if first:
        return True
    try:
        return math.isclose(
            float(candidate.strip().split()[0]),
            float(expected.strip()),
            rel_tol=_REL_TOL,
        )
    except Exception:  # noqa: BLE001 - official parser treats it as a miss
        return False


def evaluate(response: str, problem: ProblemSpec, answer: AnswerSpec) -> dict[str, Any]:
    candidate = _boxed(response)
    expected = answer.number
    if _scientific_unit(answer.unit):
        candidate = _scale(_notation(candidate)) if candidate is not None else None
        expected = _scale((expected, answer.unit))
    passed = _equivalent(candidate, expected)
    return {
        "reward": float(passed),
        "mode": "scibench",
        "status": "passed" if passed else "different",
        "split": problem.split,
        "textbook": problem.textbook,
        "prediction": candidate,
    }
