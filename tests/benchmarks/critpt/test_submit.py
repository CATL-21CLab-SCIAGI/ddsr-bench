import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import httpx
import pytest

from ddsr_bench.benchmarks.critpt import submission
from ddsr_bench.benchmarks.critpt.submission.official import (
    build_batch,
    submit_official,
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
    result = submit_official(
        {"submissions": submissions, "batch_metadata": {}},
        "secret",
        transport=httpx.MockTransport(respond),
    )

    assert calls == 1
    assert result == {"accuracy": 0.5}

    with pytest.raises(ValueError, match="refusing"):
        submit_official(
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

    result = submit_official(payload, "secret", transport=httpx.MockTransport(respond))
    assert requests == [payload]
    assert result == {"accuracy": 0.2}
    assert (
        sum(s["problem_id"] == "Challenge_1_main" for s in payload["submissions"]) == 5
    )


@pytest.mark.parametrize("attempts", [-1, [], [0, 0], [-1], [True], [0.0], ["0"]])
def test_invalid_selection(tmp_path: Path, attempts: list) -> None:
    from ddsr_bench.benchmarks.critpt.submission import submit

    with pytest.raises(ValueError):
        submit(tmp_path, {"attempts": attempts})


@pytest.mark.parametrize("attempts", [None, True, 0.0, "0", {"0": True}])
def test_selection_type(tmp_path: Path, attempts) -> None:
    from ddsr_bench.benchmarks.critpt.submission import submit

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
        submit_official(payload, "secret", transport=httpx.MockTransport(respond))
    assert calls == []


def test_unbalanced_attempts(tmp_path: Path) -> None:
    make_job(tmp_path)
    payload = build_batch(tmp_path, [0])
    payload["submissions"].append(payload["submissions"][0])
    with pytest.raises(ValueError, match="incomplete"):
        submit_official(payload, "secret")


@pytest.mark.parametrize("collected", [False, True])
@pytest.mark.parametrize("count", [1, 70])
def test_collection(tmp_path, monkeypatch, collected, count):
    trials = make_job(tmp_path)[:count]
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"batches": [{"attempt": 0, "trials": trials}]}))
    before = summary.read_bytes()
    if collected:
        monkeypatch.setattr(
            submission, "collect_trials", lambda _: pytest.fail("already collected")
        )
    else:
        summary.unlink()
        for trial in trials:
            (tmp_path / trial["trial_name"] / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": f"critpt/{trial['problem_id']}",
                        "trial_name": trial["trial_name"],
                        "attempt": 0,
                        "agent_info": {
                            "name": "critpt",
                            "model_info": {"name": "model"},
                        },
                        "static_result": {"status": "validated"},
                    }
                )
            )
    calls = []
    monkeypatch.setenv("TEST_AA_KEY", "fake")
    monkeypatch.setattr(
        submission,
        "submit_official",
        lambda payload, *a, **k: calls.append(payload) or {},
    )
    config = {
        "attempts": 0,
        "api_key_env": "TEST_AA_KEY",
        "endpoint": "unused",
        "timeout_sec": 1,
    }
    if count == 70:
        submission.submit(tmp_path, config)
        assert len(calls) == 1
        assert len(calls[0]["submissions"]) == 70
    else:
        with pytest.raises(ValueError, match="70 unique"):
            submission.submit(tmp_path, config)
        assert not calls
        assert not list(tmp_path.glob("submission*.json"))
    if collected:
        assert summary.read_bytes() == before
    else:
        assert summary.is_file()
        assert (tmp_path / "summary.csv").is_file()


@pytest.fixture
def selected_job(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_AA_KEY", "fake")
    monkeypatch.setattr(submission, "collect_trials", lambda _: {})
    config = {
        "attempts": [1, 0],
        "api_key_env": "TEST_AA_KEY",
        "endpoint": "unused",
        "timeout_sec": 1,
    }
    monkeypatch.setattr(
        submission, "build_batch", lambda _, attempts: {"attempts": attempts}
    )
    return config


def test_duplicate_selection(tmp_path, monkeypatch, selected_job):
    calls = []
    monkeypatch.setattr(
        submission,
        "submit_official",
        lambda payload, *args, **kwargs: calls.append(payload) or {},
    )
    submission.submit(tmp_path, selected_job)
    with pytest.raises(ValueError, match="already exists"):
        submission.submit(tmp_path, selected_job | {"attempts": [0, 1]})
    assert calls == [{"attempts": [0, 1]}]
    assert not list(tmp_path.glob("*.pending.json"))


def test_concurrent_selection(tmp_path, monkeypatch, selected_job):
    barrier = Barrier(2, timeout=5)
    calls = []

    def build(*args):
        barrier.wait()  # Both callers passed the initial existence check.
        return {}

    def submit(_):
        try:
            submission.submit(tmp_path, selected_job)
            return "sent"
        except ValueError as error:
            assert "already" in str(error)
            return "blocked"

    monkeypatch.setattr(submission, "build_batch", build)
    monkeypatch.setattr(
        submission, "submit_official", lambda *a, **k: calls.append(True) or {}
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit, range(2))) == ["blocked", "sent"]
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["request", "save"])
def test_uncertain_submission(tmp_path, monkeypatch, selected_job, failure):
    calls = []
    write = submission.write_json

    def send(*args, **kwargs):
        calls.append(True)
        if failure == "request":
            raise TimeoutError("unknown AA outcome")
        return {}

    def save(path, *args, **kwargs):
        if failure == "save" and path.name == "submission-0-1.json":
            raise OSError("disk full")
        return write(path, *args, **kwargs)

    monkeypatch.setattr(submission, "submit_official", send)
    monkeypatch.setattr(submission, "write_json", save)
    with pytest.raises(OSError):
        submission.submit(tmp_path, selected_job)
    assert (tmp_path / "submission-0-1.pending.json").is_file()
    with pytest.raises(ValueError, match="already pending"):
        submission.submit(tmp_path, selected_job)
    assert calls == [True]


def test_failed_preflight(tmp_path, monkeypatch, selected_job):
    monkeypatch.setattr(
        submission, "submit_official", lambda *a, **k: pytest.fail("must not send")
    )
    monkeypatch.delenv("TEST_AA_KEY")
    with pytest.raises(ValueError, match="not set"):
        submission.submit(tmp_path, selected_job)
    assert not list(tmp_path.iterdir())
