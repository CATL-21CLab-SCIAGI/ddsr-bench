from __future__ import annotations

import asyncio
import json
from typing import Any

from ddsr_bench.benchmarks.physics.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.benchmarks.physics.generation.prompts import judge_prompt
from ddsr_bench.generation.client import ChatClient


def final_text(response: str) -> str:
    """Drop a reasoning block leaked into visible response content upstream-style."""
    return response.split("</think>")[-1].strip()


def parse_judgment(raw: str, count: int) -> dict[str, Any] | None:
    """Validate the official judge object, returning None when it needs a retry."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, list) or len(answers) != count:
        return None
    if any(
        not isinstance(answer, dict)
        or str(answer.get("correct", "")).strip().lower() not in ("yes", "no")
        for answer in answers
    ):
        return None
    try:
        confidence = max(
            0,
            min(
                100,
                round(float(str(data.get("confidence", 100)).rstrip("%"))),
            ),
        )
    except ValueError:
        confidence = 100
    return {
        "answers": [
            {
                "index": index,
                "extracted_final_answer": answer.get("extracted_final_answer"),
                "reasoning": answer.get("reasoning"),
                "correct": str(answer["correct"]).strip().lower() == "yes",
            }
            for index, answer in enumerate(answers)
        ],
        "confidence": confidence,
    }


def unanswered(count: int, reason: str) -> dict[str, Any]:
    return {
        "confidence": 100,
        "answers": [
            {
                "index": index,
                "extracted_final_answer": None,
                "reasoning": reason,
                "correct": False,
            }
            for index in range(count)
        ],
    }


async def evaluate(
    response: str,
    problem: ProblemSpec,
    answer: AnswerSpec,
    client: ChatClient,
    *,
    max_tokens: int = 32_768,
    retries: int = 4,
) -> dict[str, Any]:
    """Apply the pinned PHYSICS-Verified judge protocol to one response."""
    visible = final_text(response)
    if not visible:
        judgment = unanswered(len(answer.final_answers), "no final answer")
        return _result(problem, answer, judgment, skipped="no final answer")

    request = ({"role": "user", "content": judge_prompt(problem, visible, answer)},)
    for attempt in range(retries):
        try:
            response = await client.chat(
                request,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            judgment = parse_judgment(response.content, len(answer.final_answers))
            if judgment is not None:
                return _result(problem, answer, judgment, attempts=attempt + 1)
        except Exception:  # noqa: BLE001, S110 - official protocol retries failures
            pass
        await asyncio.sleep(2 * (attempt + 1))
    return {
        "reward": 0.0,
        "mode": "physics",
        "status": "error",
        "error": "JudgeError",
        "judged": False,
        "split": problem.split,
        "domain": problem.domain,
        "answer_types": list(answer.types),
        "answer_correct": [False] * len(answer.final_answers),
        "confidence": 100,
    }


def _result(
    problem: ProblemSpec,
    answer: AnswerSpec,
    judgment: dict[str, Any],
    **details: Any,
) -> dict[str, Any]:
    flags = [item["correct"] for item in judgment["answers"]]
    passed = all(flags)
    return {
        "reward": float(passed),
        "mode": "physics",
        "status": "passed" if passed else "different",
        "judged": True,
        "split": problem.split,
        "domain": problem.domain,
        "answer_types": list(answer.types),
        "answer_correct": flags,
        "confidence": judgment["confidence"],
        "judgment": judgment,
        **details,
    }
