import json

import pytest

from ddsr_bench.benchmarks.physics.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.benchmarks.physics.evaluation.evaluator import (
    evaluate,
    final_text,
    parse_judgment,
)
from ddsr_bench.benchmarks.physics.evaluation.metrics import (
    calibration_error,
    score_results,
)
from ddsr_bench.generation.client import ChatResponse

PROBLEM = ProblemSpec("p1", "test", "Optics", "Q?", ())
ANSWER = AnswerSpec(
    "solution", ("x = 1", "y = 2"), ("(a)", "(b)"), ("numeric", "numeric")
)


class FakeJudge:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return ChatResponse(self.content, None, "judge", None, None, 0.1, {}, "stop")


def test_parser_and_visible_text() -> None:
    raw = json.dumps(
        {
            "answers": [{"correct": "yes"}, {"correct": "No"}],
            "confidence": "85%",
        }
    )

    judgment = parse_judgment(raw, 2)

    assert [answer["correct"] for answer in judgment["answers"]] == [True, False]
    assert judgment["confidence"] == 85
    assert parse_judgment(raw, 3) is None
    assert parse_judgment("not json", 2) is None
    assert final_text("<think>hidden</think>\nAnswers: 1") == "Answers: 1"


@pytest.mark.asyncio
async def test_evaluate() -> None:
    judge = FakeJudge(
        json.dumps(
            {
                "answers": [{"correct": "yes"}, {"correct": "no"}],
                "confidence": 80,
            }
        )
    )

    result = await evaluate("Answers: x=1, y=3", PROBLEM, ANSWER, judge)

    assert result["reward"] == 0
    assert result["status"] == "different"
    assert result["answer_correct"] == [True, False]
    assert result["answer_types"] == ["numeric", "numeric"]
    assert judge.calls[0][1] == {
        "max_tokens": 32768,
        "response_format": {"type": "json_object"},
    }


@pytest.mark.asyncio
async def test_empty_response_skips_judge() -> None:
    judge = FakeJudge("unused")

    result = await evaluate("<think>unfinished</think>", PROBLEM, ANSWER, judge)

    assert result["answer_correct"] == [False, False]
    assert result["skipped"] == "no final answer"
    assert judge.calls == []


def test_official_metrics() -> None:
    results = [
        {
            "answer_correct": [True, False],
            "confidence": 90,
            "status": "different",
            "judged": True,
            "domain": "Optics",
            "answer_types": ["numeric", "numeric"],
        },
        {
            "answer_correct": [False],
            "confidence": 100,
            "status": "error",
            "judged": False,
            "domain": "Optics",
            "answer_types": ["symbolic"],
        },
    ]

    metrics = score_results(results, repetitions=50)

    assert metrics["strict_accuracy"] == 0
    assert metrics["answer_accuracy"] == 1 / 3
    assert metrics["partial_credit"] == 0.25
    assert metrics["unjudged"] == 1
    assert metrics["answer_accuracy_by_type"] == {"numeric": 0.5, "symbolic": 0}
    assert calibration_error([0.5] * 10, [True] * 10) is None
    assert calibration_error(
        [0.9] * 100 + [1.0] * 100, [False] * 100 + [True] * 100
    ) == pytest.approx((0.5 * 0.81) ** 0.5)


def test_synthetic_unanswered_judgment_counts_as_judged() -> None:
    result = {
        "answer_correct": [False],
        "confidence": 100,
        "status": "error",
        "judged": True,
        "domain": "Optics",
        "answer_types": ["numeric"],
    }

    assert score_results([result], repetitions=10)["unjudged"] == 0
