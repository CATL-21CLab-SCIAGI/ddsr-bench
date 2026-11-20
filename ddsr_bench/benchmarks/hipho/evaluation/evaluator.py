from __future__ import annotations

import asyncio
from typing import Any

from ddsr_bench.benchmarks.hipho.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.generation.client import ChatClient

from .official import shim


class _Judge:
    """Bridge VLMEvalKit's synchronous judge calls to the async shared client."""

    def __init__(self, client: ChatClient, loop: asyncio.AbstractEventLoop, limit: int):
        self.client = client
        self.loop = loop
        self.limit = limit
        self.calls = 0

    def generate(self, prompt: str, **_: Any) -> str:
        future = asyncio.run_coroutine_threadsafe(
            self.client.chat(
                ({"role": "user", "content": prompt},), max_tokens=self.limit
            ),
            self.loop,
        )
        self.calls += 1
        return future.result().content


def _score(
    scorer: Any,
    judge: _Judge,
    response: str,
    problem: ProblemSpec,
    answer: AnswerSpec,
) -> dict[str, Any]:
    """Call the two maintained scorer branches after typed source loading."""
    fine, details = scorer._evaluate_fine_grained(
        response,
        list(answer.marking),
        list(answer.points),
        judge,
        problem.question,
    )
    coarse_error = None
    try:
        coarse, _ = scorer._evaluate_coarse_grained(
            response,
            list(answer.answers),
            list(answer.answer_types),
            list(answer.units),
            list(answer.points),
            problem.question,
            judge,
        )
    except Exception as error:  # noqa: BLE001 - later extension has malformed units
        coarse, coarse_error = 0.0, type(error).__name__
    return {
        "fine": float(fine),
        "coarse": float(coarse),
        "details": details,
        "coarse_error": coarse_error,
    }


def unanswered(problem: ProblemSpec, reason: str) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "mode": "hipho",
        "status": "error",
        "error": reason,
        "judged": False,
        "points": 0.0,
        "max_points": 0.0,
        "fine_grained_points": 0.0,
        "coarse_grained_points": 0.0,
        "exam": problem.exam,
        "field": problem.field,
        "modality": problem.modality,
    }


async def evaluate(
    response: str,
    problem: ProblemSpec,
    answer: AnswerSpec,
    client: ChatClient,
    *,
    max_tokens: int = 4_096,
) -> dict[str, Any]:
    """Run VLMEvalKit's official fine/coarse scorer for one attempt."""
    if not response.strip():
        result = unanswered(problem, "NoResponse")
        result["max_points"] = sum(answer.points)
        return result
    scorer, _ = shim.new_scorer()
    judge = _Judge(client, asyncio.get_running_loop(), max_tokens)
    try:
        scored = await asyncio.to_thread(
            _score,
            scorer,
            judge,
            response,
            problem,
            answer,
        )
    except Exception as error:  # noqa: BLE001 - official scorer errors are trials
        result = unanswered(problem, type(error).__name__)
        result["max_points"] = sum(answer.points)
        return result
    points = max(scored["fine"], scored["coarse"])
    maximum = sum(answer.points)
    passed = abs(points - maximum) < 1e-8
    details = scored["details"] or []
    result = {
        "reward": points / maximum,
        "mode": "hipho",
        "status": "passed" if passed else "different",
        "judged": True,
        "points": points,
        "max_points": maximum,
        "fine_grained_points": scored["fine"],
        "coarse_grained_points": scored["coarse"],
        "marking_scores": [item.get("score") for item in details],
        "judge_calls": judge.calls,
        "exam": problem.exam,
        "field": problem.field,
        "modality": problem.modality,
    }
    if scored["coarse_error"] is not None:
        result["coarse_error"] = scored["coarse_error"]
    return result
