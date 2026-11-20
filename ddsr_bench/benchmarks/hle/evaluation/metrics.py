from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def calibration_error(
    confidence: Sequence[float], correct: Sequence[bool], beta: int = 100
) -> float | None:
    """Compute the official HLE RMS calibration error, including its last-bin bug."""
    pairs = sorted(zip(confidence, correct, strict=True))
    count = len(pairs)
    if count < beta:
        return None
    bins = [[index * beta, (index + 1) * beta] for index in range(count // beta)]
    bins[-1][1] = count
    total = 0.0
    for low, high in bins[:-1]:
        values = pairs[low:high]
        mean_confidence = sum(value for value, _ in values) / len(values)
        mean_correct = sum(value for _, value in values) / len(values)
        total += len(values) / count * (mean_confidence - mean_correct) ** 2
    return total**0.5


def score_results(results: Sequence[dict[str, Any]], beta: int = 100) -> dict[str, Any]:
    """Aggregate complete HLE trials with the official accuracy and calibration."""
    correct = [bool(result["correct"]) for result in results]
    judged = [result for result in results if result.get("judged", False)]
    confidence = [float(result["confidence"]) / 100 for result in judged]
    count = len(results)
    accuracy = 100 * sum(correct) / count
    calibration = calibration_error(
        confidence, [bool(result["correct"]) for result in judged], beta
    )
    return {
        "accuracy": round(accuracy, 2),
        "confidence_half_width": round(
            1.96 * math.sqrt(accuracy * (100 - accuracy) / count), 2
        ),
        "calibration_error": (
            None if calibration is None else 100 * round(calibration, 2)
        ),
        "problems": count,
        "unjudged": sum(not result.get("judged", False) for result in results),
    }
