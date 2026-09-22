import json
from pathlib import Path

import httpx
import pytest

from ddsr_bench.benchmarks.critpt.evaluation.submission.official import (
    build_batch,
    submit_batch,
)


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
                "benchmark_config": {"strategy": "one-step"},
                "trial_name": trial_name,
                "artifact": f"{trial_name}/artifacts/answer.py",
                "attempt": 0,
            }
        )
    (path / "summary.json").write_text(
        json.dumps({"batches": [{"attempt": 0, "trials": trials}]})
    )
    return trials


def test_builds_official_batch(tmp_path: Path) -> None:
    make_job(tmp_path)

    payload = build_batch(tmp_path, [0])

    assert len(payload["submissions"]) == 70
    assert payload["submissions"][0]["problem_id"] == "Challenge_1_main"
    assert payload["submissions"][0]["generated_code"].startswith("```python")
    assert set(payload["submissions"][0]) == {
        "problem_id",
        "generated_code",
        "model",
        "generation_config",
        "messages",
    }
    assert payload["submissions"][0]["messages"] == [
        {"role": "user", "content": "problem"}
    ]
    assert payload["batch_metadata"] == {"attempts": [0]}


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
        build_batch(tmp_path, [0])


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


def test_preflight_five_attempts(tmp_path: Path) -> None:
    trials = make_job(tmp_path)
    batches = [
        {"attempt": attempt, "trials": [dict(t, attempt=attempt) for t in trials]}
        for attempt in range(5)
    ]
    (tmp_path / "summary.json").write_text(json.dumps({"batches": batches}))

    payload = build_batch(tmp_path, [0, 1, 2, 3, 4])

    assert len(payload["submissions"]) == 350
    assert payload["batch_metadata"]["attempts"] == list(range(5))
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"accuracy": 0.2})

    result = submit_batch(payload, "secret", transport=httpx.MockTransport(respond))
    assert requests == [payload]
    assert result == {"accuracy": 0.2}
    assert (
        sum(s["problem_id"] == "Challenge_1_main" for s in payload["submissions"]) == 5
    )


@pytest.mark.parametrize("attempts", [-1, [], [0, 0], [-1], [True], [0.0], ["0"]])
def test_invalid_selection(tmp_path: Path, attempts: list) -> None:
    from ddsr_bench.benchmarks.critpt.evaluation.submission import submit

    with pytest.raises(ValueError):
        submit(tmp_path, {"attempts": attempts})


@pytest.mark.parametrize("attempts", [None, True, 0.0, "0", {"0": True}])
def test_selection_type(tmp_path: Path, attempts) -> None:
    from ddsr_bench.benchmarks.critpt.evaluation.submission import submit

    with pytest.raises(TypeError, match="must be an integer or list"):
        submit(tmp_path, {"attempts": attempts})


def test_preflight_before_send(tmp_path: Path) -> None:
    make_job(tmp_path)
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    # Attempt 0 is valid, but absent attempt 1 must abort the entire selection.
    with pytest.raises(ValueError, match="attempt 1"):
        payload = build_batch(tmp_path, [0, 1])
        submit_batch(payload, "secret", transport=httpx.MockTransport(respond))
    assert calls == []


def test_unbalanced_attempts(tmp_path: Path) -> None:
    make_job(tmp_path)
    payload = build_batch(tmp_path, [0])
    payload["submissions"].append(payload["submissions"][0])
    with pytest.raises(ValueError, match="incomplete"):
        submit_batch(payload, "secret")
