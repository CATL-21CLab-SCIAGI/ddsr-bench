from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

ENDPOINT = "https://artificialanalysis.ai/api/v2/critpt/evaluate"
OFFICIAL_IDS = {f"Challenge_{number}_main" for number in range(1, 71)}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def build_batch(job_dir: str | Path, attempt: int) -> dict[str, Any]:
    """Build one explicit official batch from a collected attempt."""
    job = Path(job_dir)
    summary = _read(job / "summary.json")
    batches = summary.get("batches")
    if not isinstance(batches, list) or any(
        not isinstance(batch, dict) for batch in batches
    ):
        raise TypeError("summary batches must be a list of objects")
    selected = [batch for batch in batches if batch.get("attempt") == attempt]
    if len(selected) != 1 or not isinstance(selected[0].get("trials"), list):
        raise ValueError(f"summary has no unique attempt {attempt}")
    trials = selected[0]["trials"]
    if any(not isinstance(trial, dict) for trial in trials):
        raise TypeError("summary trials must be objects")
    ids = [trial.get("problem_id") for trial in trials]
    if len(ids) != 70 or len(set(ids)) != 70 or set(ids) != OFFICIAL_IDS:
        raise ValueError("official submission requires exactly 70 unique problem IDs")
    if any(trial.get("attempt") != attempt for trial in trials):
        raise ValueError("official submission cannot mix attempt indices")

    submissions = []
    ordered = sorted(trials, key=lambda item: int(item["problem_id"].split("_")[1]))
    for trial in ordered:
        answer = trial.get("answer")
        if not isinstance(answer, str) or not (job / answer).is_file():
            raise ValueError(f"{trial['problem_id']} has no answer artifact")
        record = _read(job / trial["trial_name"] / "agent" / "response.json")
        responses = record.get("responses")
        messages = record.get("messages")
        if not isinstance(responses, list) or not responses:
            raise ValueError(f"{trial['problem_id']} has no model response")
        content = responses[-1].get("content")
        if not isinstance(content, str) or not content:
            raise ValueError(f"{trial['problem_id']} has no generated code")
        if not isinstance(messages, list) or any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError(f"{trial['problem_id']} has invalid messages")
        model = record.get("model") or trial.get("model")
        strategy = record.get("strategy")
        if not isinstance(model, str) or not isinstance(strategy, str):
            raise TypeError(f"{trial['problem_id']} has invalid generation metadata")
        submissions.append(
            {
                "problem_id": trial["problem_id"],
                "generated_code": content,
                "model": model,
                "generation_config": {
                    "strategy": strategy,
                    "seed": record.get("seed"),
                },
                "messages": [
                    {"role": message["role"], "content": message["content"]}
                    for message in messages
                ],
            }
        )
    return {
        "submissions": submissions,
        "batch_metadata": {
            "attempt": attempt,
            "agent": trials[0]["agent"],
            "strategy": trials[0]["strategy"],
        },
    }


def submit_batch(
    payload: dict[str, Any],
    api_key: str,
    *,
    endpoint: str = ENDPOINT,
    timeout: float = 7_200,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Submit once; intentionally do not retry a rate-limited operation."""
    if not api_key:
        raise ValueError("an Artificial Analysis API key is required")
    submissions = payload.get("submissions")
    if not isinstance(submissions, list) or any(
        not isinstance(item, dict) for item in submissions
    ):
        raise TypeError("submissions must be a list of objects")
    ids = [item.get("problem_id") for item in submissions]
    if len(ids) != 70 or len(set(ids)) != 70 or set(ids) != OFFICIAL_IDS:
        raise ValueError("refusing to submit an incomplete official batch")
    with httpx.Client(timeout=timeout, transport=transport) as client:
        response = client.post(endpoint, json=payload, headers={"x-api-key": api_key})
        response.raise_for_status()
    try:
        result = response.json()
    except ValueError as error:
        raise ValueError("grading server returned invalid JSON") from error
    if not isinstance(result, dict):
        raise TypeError("grading server response must be a JSON object")
    return result
