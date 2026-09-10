import json
from pathlib import Path

import httpx
import pytest

from critpt_eval.benchmark.submit import build_batch, submit_batch


def make_job(path: Path) -> list[dict]:
    trials = []
    for number in range(1, 71):
        problem_id = f"Challenge_{number}_main"
        trial_name = f"trial-{number}"
        trial = path / trial_name
        (trial / "agent").mkdir(parents=True)
        (trial / "artifacts").mkdir()
        (trial / "artifacts" / "answer.py").write_text("def answer():\n    return 1\n")
        (trial / "agent" / "response.json").write_text(
            json.dumps(
                {
                    "model": "model",
                    "strategy": "one-step",
                    "seed": 7,
                    "responses": [{"content": "```python\nanswer\n```"}],
                    "messages": [
                        {"role": "user", "content": "problem", "sha256": "hash"}
                    ],
                }
            )
        )
        trials.append(
            {
                "problem_id": problem_id,
                "agent": "critpt",
                "model": "model",
                "strategy": "one-step",
                "trial_name": trial_name,
                "answer": f"{trial_name}/artifacts/answer.py",
                "attempt": 0,
            }
        )
    (path / "summary.json").write_text(
        json.dumps({"batches": [{"attempt": 0, "trials": trials}]})
    )
    return trials


def test_builds_official_batch(tmp_path: Path) -> None:
    make_job(tmp_path)

    payload = build_batch(tmp_path, 0)

    assert len(payload["submissions"]) == 70
    assert payload["submissions"][0]["problem_id"] == "Challenge_1_main"
    assert payload["submissions"][0]["generated_code"].startswith("```python")
    assert payload["submissions"][0]["messages"] == [
        {"role": "user", "content": "problem"}
    ]
    assert payload["batch_metadata"] == {
        "attempt": 0,
        "agent": "critpt",
        "strategy": "one-step",
    }


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "mixed"])
def test_rejects_invalid_batches(tmp_path: Path, invalid: str) -> None:
    trials = make_job(tmp_path)
    if invalid == "missing":
        trials.pop()
    elif invalid == "duplicate":
        trials[-1]["problem_id"] = trials[0]["problem_id"]
    else:
        trials[-1]["attempt"] = 1
    (tmp_path / "summary.json").write_text(
        json.dumps({"batches": [{"attempt": 0, "trials": trials}]})
    )

    with pytest.raises(ValueError):
        build_batch(tmp_path, 0)


def test_submits_once() -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["x-api-key"] == "secret"
        return httpx.Response(200, json={"accuracy": 0.5})

    submissions = [
        {"problem_id": f"Challenge_{number}_main"} for number in range(1, 71)
    ]
    result = submit_batch(
        {"submissions": submissions, "batch_metadata": {}},
        "secret",
        transport=httpx.MockTransport(respond),
    )

    assert calls == 1
    assert result == {"accuracy": 0.5}

    with pytest.raises(ValueError, match="refusing"):
        submit_batch(
            {"submissions": [], "batch_metadata": {}},
            "secret",
            transport=httpx.MockTransport(respond),
        )
    assert calls == 1
