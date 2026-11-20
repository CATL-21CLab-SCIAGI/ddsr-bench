from __future__ import annotations

import re
from typing import Any

from ddsr_bench.benchmarks.frontierscience.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.benchmarks.frontierscience.generation.prompts import judge_prompt
from ddsr_bench.generation.client import ChatClient

_OLYMPIAD = re.compile(r"GRADE:\s*([CI])\s*$", re.IGNORECASE)
_RESEARCH = re.compile(r"VERDICT:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE)


def parse_judgment(track: str, response: str) -> tuple[float, bool] | None:
    """Parse the final official judge marker without repairing it."""
    match = (_OLYMPIAD if track == "olympiad" else _RESEARCH).search(response)
    if match is None:
        return None
    if track == "olympiad":
        passed = match.group(1).upper() == "C"
        return float(passed), passed
    points = min(10.0, max(0.0, float(match.group(1))))
    return points, points >= 7


def unanswered(problem: ProblemSpec, reason: str) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "mode": "frontierscience",
        "status": "error",
        "error": reason,
        "judged": False,
        "track": problem.track,
        "subject": problem.subject,
        "correct": False,
        "points": 0.0,
    }


async def evaluate(
    response: str,
    problem: ProblemSpec,
    answer: AnswerSpec,
    client: ChatClient,
    *,
    max_tokens: int = 4_096,
) -> dict[str, Any]:
    """Apply the official judge protocol for the problem's track."""
    if not response.strip():
        return unanswered(problem, "NoResponse")
    request = ({"role": "user", "content": judge_prompt(problem, response, answer)},)
    try:
        judged = await client.chat(request, max_tokens=max_tokens)
    except Exception as error:  # noqa: BLE001 - judge failure is an unscored trial
        return unanswered(problem, type(error).__name__)
    parsed = parse_judgment(problem.track, judged.content)
    if parsed is None:
        return unanswered(problem, "JudgeFormatError")
    points, correct = parsed
    reward = points if problem.track == "olympiad" else points / 10
    return {
        "reward": reward,
        "mode": "frontierscience",
        "status": "passed" if correct else "different",
        "judged": True,
        "track": problem.track,
        "subject": problem.subject,
        "correct": correct,
        "points": points,
        "judge_response": judged.content,
    }
