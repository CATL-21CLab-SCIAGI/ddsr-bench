from __future__ import annotations

import json
from typing import Any

from ddsr_bench.benchmarks.hle.data.schemas import AnswerSpec, ProblemSpec
from ddsr_bench.benchmarks.hle.generation.prompts import judge_prompt
from ddsr_bench.generation.client import ChatClient


def parse_judgment(raw: str) -> dict[str, Any] | None:
    """Validate the fields retained from the official structured judge output."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if (
        not isinstance(data, dict)
        or data.get("correct") not in ("yes", "no")
        or ("strict" in data and data["strict"] is not True)
    ):
        return None
    extracted = data.get("extracted_final_answer")
    reasoning = data.get("reasoning")
    confidence = data.get("confidence")
    if (
        not isinstance(extracted, str)
        or not isinstance(reasoning, str)
        or type(confidence) is not int
        or not 0 <= confidence <= 100
    ):
        return None
    return {
        "extracted_final_answer": extracted,
        "reasoning": reasoning,
        "correct": data["correct"] == "yes",
        "confidence": confidence,
    }


def unanswered(reason: str) -> dict[str, Any]:
    return {
        "extracted_final_answer": "None",
        "reasoning": reason,
        "correct": False,
        "confidence": 100,
    }


async def evaluate(
    response: str,
    problem: ProblemSpec,
    answer: AnswerSpec,
    client: ChatClient,
    *,
    max_tokens: int = 4_096,
) -> dict[str, Any]:
    """Apply the pinned HLE judge protocol to one response."""
    request = ({"role": "user", "content": judge_prompt(problem, response, answer)},)
    try:
        judged = await client.chat(
            request,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        judgment = parse_judgment(judged.content)
        if judgment is not None:
            return _result(problem, answer, judgment)
    except Exception:  # noqa: BLE001, S110 - judge failures receive zero
        pass
    return {
        "reward": 0.0,
        "mode": "hle",
        "status": "error",
        "error": "JudgeError",
        "judged": False,
        "subset": problem.subset,
        "category": problem.category,
        "answer_type": answer.type,
        "correct": False,
        "confidence": 100,
    }


def _result(
    problem: ProblemSpec,
    answer: AnswerSpec,
    judgment: dict[str, Any],
    **details: Any,
) -> dict[str, Any]:
    passed = judgment["correct"]
    return {
        "reward": float(passed),
        "mode": "hle",
        "status": "passed" if passed else "different",
        "judged": True,
        "subset": problem.subset,
        "category": problem.category,
        "answer_type": answer.type,
        "correct": passed,
        "confidence": judgment["confidence"],
        "judgment": judgment,
        **details,
    }
