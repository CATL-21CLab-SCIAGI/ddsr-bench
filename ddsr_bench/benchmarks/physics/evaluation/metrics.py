from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def calibration_error(
    confidence: Sequence[float], correct: Sequence[bool], beta: int = 100
) -> float | None:
    """Compute the HLE RMS calibration error, including its skipped last bin."""
    pairs = sorted(zip(confidence, correct, strict=True))
    count = len(pairs)
    if count < beta:
        return None
    bins = [[index * beta, (index + 1) * beta] for index in range(count // beta)]
    bins[-1][1] = count
    total = 0.0
    for low, high in bins[:-1]:
        confidences = [value for value, _ in pairs[low:high]]
        correctness = [value for _, value in pairs[low:high]]
        total += (
            len(confidences)
            / count
            * (
                sum(confidences) / len(confidences)
                - sum(correctness) / len(correctness)
            )
            ** 2
        )
    return total**0.5


def _rates(problems: Sequence[Sequence[bool]]) -> dict[str, float]:
    flags = [flag for problem in problems for flag in problem]
    return {
        "strict_accuracy": sum(all(problem) for problem in problems) / len(problems),
        "answer_accuracy": sum(flags) / len(flags),
        "partial_credit": sum(sum(problem) / len(problem) for problem in problems)
        / len(problems),
    }


def _bootstrap(
    problems: Sequence[Sequence[bool]], repetitions: int, seed: int = 0
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    samples = [
        _rates([problems[rng.randrange(len(problems))] for _ in problems])
        for _ in range(repetitions)
    ]
    return {
        key: [
            sorted(sample[key] for sample in samples)[int(0.025 * repetitions)],
            sorted(sample[key] for sample in samples)[int(0.975 * repetitions) - 1],
        ]
        for key in samples[0]
    }


def score_results(
    results: Sequence[dict[str, Any]], repetitions: int = 2_000
) -> dict[str, Any]:
    """Aggregate saved trials with the official PHYSICS-Verified metrics."""
    problems = [result["answer_correct"] for result in results]
    confidence = [result["confidence"] / 100 for result in results]
    output: dict[str, Any] = _rates(problems)
    output["ci95"] = _bootstrap(problems, repetitions)
    output["calibration_error"] = calibration_error(
        confidence, [all(problem) for problem in problems]
    )
    output["problems"] = len(problems)
    output["answers"] = sum(map(len, problems))
    output["unjudged"] = sum(
        not result.get("judged", result["status"] != "error") for result in results
    )

    by_domain: dict[str, list[list[bool]]] = defaultdict(list)
    by_type: dict[str, list[bool]] = defaultdict(list)
    for result, flags in zip(results, problems, strict=True):
        by_domain[result["domain"]].append(flags)
        for answer_type, flag in zip(result["answer_types"], flags, strict=True):
            by_type[answer_type].append(flag)
    output["strict_accuracy_by_domain"] = {
        domain: _rates(values)["strict_accuracy"]
        for domain, values in sorted(by_domain.items())
    }
    output["answer_accuracy_by_type"] = {
        answer_type: sum(values) / len(values)
        for answer_type, values in sorted(by_type.items())
    }
    return output
