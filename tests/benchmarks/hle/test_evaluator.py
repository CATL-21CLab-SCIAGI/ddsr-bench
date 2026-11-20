import json

import pytest

from ddsr_bench.benchmarks.hle.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.benchmarks.hle.evaluation.evaluator import evaluate, parse_judgment
from ddsr_bench.benchmarks.hle.evaluation.metrics import (
    calibration_error,
    score_results,
)
from ddsr_bench.generation.client import ChatResponse

PROBLEM = ProblemSpec("p1", "Gold", "Physics", "Mechanics", "Q?", None)
ANSWER = AnswerSpec("42", "exactMatch", "because", None)


class FakeJudge:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return ChatResponse(self.content, None, "judge", None, None, 0.1, {}, "stop")


def test_parser() -> None:
    raw = json.dumps(
        {
            "extracted_final_answer": "42",
            "reasoning": "matches",
            "correct": "yes",
            "confidence": 80,
        }
    )
    assert parse_judgment(raw)["correct"] is True
    assert parse_judgment(raw)["confidence"] == 80
    assert parse_judgment("not json") is None
    assert parse_judgment(raw.replace('"yes"', '"maybe"')) is None


@pytest.mark.asyncio
async def test_evaluate() -> None:
    judge = FakeJudge(
        json.dumps(
            {
                "extracted_final_answer": "41",
                "reasoning": "different",
                "correct": "no",
                "confidence": 70,
            }
        )
    )
    result = await evaluate("Answer: 41", PROBLEM, ANSWER, judge)

    assert result["reward"] == 0
    assert result["status"] == "different"
    assert judge.calls[0][1] == {
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }


@pytest.mark.asyncio
async def test_malformed_judgment_is_not_repaired() -> None:
    judge = FakeJudge("not json")

    result = await evaluate("Answer: 42", PROBLEM, ANSWER, judge)

    assert result["status"] == "error"
    assert result["judged"] is False
    assert len(judge.calls) == 1


def test_official_metrics() -> None:
    results = [
        {"correct": True, "confidence": 90, "judged": True},
        {"correct": False, "confidence": 100, "judged": False},
    ]
    metrics = score_results(results)
    assert metrics["accuracy"] == 50
    assert metrics["confidence_half_width"] == 69.3
    assert metrics["calibration_error"] is None
    assert metrics["unjudged"] == 1
    assert calibration_error(
        [0.9] * 100 + [1.0] * 100, [False] * 100 + [True] * 100
    ) == pytest.approx((0.5 * 0.81) ** 0.5)


def test_unjudged_trials_do_not_enter_calibration() -> None:
    judged = {"correct": True, "confidence": 90, "judged": True}
    unjudged = {"correct": False, "confidence": 0, "judged": False}

    metrics = score_results([judged] * 100 + [unjudged] * 100)

    assert metrics["accuracy"] == 50
    assert metrics["calibration_error"] == 0
