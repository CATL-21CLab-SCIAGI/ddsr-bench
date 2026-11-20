from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def score_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate accuracy and the Research track's normalized rubric score."""
    count = len(results)
    if count == 0:
        return {}
    judged = [result for result in results if result.get("judged", False)]
    correct = sum(bool(result["correct"]) for result in results)
    accuracy = 100 * correct / count
    rewards = [float(result["reward"]) for result in results]
    return {
        "score": 100 * sum(rewards) / count,
        "accuracy": accuracy,
        "confidence_half_width": 1.96 * math.sqrt(accuracy * (100 - accuracy) / count),
        "problems": count,
        "unjudged": count - len(judged),
    }
