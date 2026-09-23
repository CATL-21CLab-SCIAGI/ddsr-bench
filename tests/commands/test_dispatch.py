import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ddsr_bench.commands.dispatch import LOGGER_NAME, configure_logging, dispatch


def test_collect_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ddsr_bench.commands.dispatch.collect_trials",
        lambda _: {"complete_batches": 2, "incomplete_batches": 1},
    )
    config = SimpleNamespace(
        action="collect", paths=SimpleNamespace(input="harbor-job")
    )

    assert dispatch(config) == "collected 2 complete and 1 incomplete batches"


def test_prepares_scicode(monkeypatch: pytest.MonkeyPatch) -> None:
    def prepare(config, source, output, resources):
        assert config.split == "validation"
        assert source is None
        assert output == "tasks/scicode-validation"
        assert resources.cpus == 2
        return [Path(output)]

    monkeypatch.setattr("ddsr_bench.commands.dispatch.preparer", lambda _: prepare)
    config = SimpleNamespace(
        action="prepare",
        benchmark=SimpleNamespace(name="scicode", split="validation"),
        paths=SimpleNamespace(input=None, output="tasks/scicode-validation"),
        harbor=SimpleNamespace(
            image="ddsr-bench-scicode:test",
            cpus=2,
            memory_mb=4096,
            timeout_sec=1800,
        ),
    )

    assert dispatch(config) == "prepared 1 Harbor tasks in tasks/scicode-validation"


def test_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="unknown action"):
        dispatch(SimpleNamespace(action="run"))


def test_export_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ddsr_bench.commands.dispatch.export_trajectories",
        lambda job, output: Path(output) / "trajectories.jsonl",
    )
    monkeypatch.setattr(
        "ddsr_bench.commands.dispatch.export_sft",
        lambda trajectories, output, view: Path(output) / f"sft-{view}.jsonl",
    )
    config = SimpleNamespace(
        action="export",
        paths=SimpleNamespace(input="job", output="dataset"),
        training=SimpleNamespace(view="answer"),
    )

    result = dispatch(config)

    assert "dataset/trajectories.jsonl" in result
    assert "dataset/sft-answer.jsonl" in result


@pytest.mark.parametrize("selection", [0, 2, [2], [4, 2, 0, 3, 1]])
def test_submit_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret")
    selected = [selection] if isinstance(selection, int) else sorted(selection)
    from ddsr_bench.benchmarks.critpt import submission

    def build(job, attempts):
        assert job == tmp_path
        assert attempts == selected
        assert isinstance(attempts, list)
        return {"attempts": attempts}

    monkeypatch.setattr(submission, "build_batch", build)
    monkeypatch.setattr(submission, "collect_trials", lambda _: {})
    monkeypatch.setattr(
        submission,
        "submit_official",
        lambda payload, *args, **kwargs: payload,
    )
    config = SimpleNamespace(
        action="submit",
        benchmark=SimpleNamespace(
            name="critpt",
            get=lambda _: {
                "endpoint": "endpoint",
                "timeout_sec": 10,
                "api_key_env": "TEST_API_KEY",
                "attempts": selection,
            },
        ),
        paths=SimpleNamespace(input=tmp_path, output=None),
    )

    result = dispatch(config)

    assert f"submitted attempts {selected}" in result
    label = "-".join(map(str, selected))
    assert json.loads((tmp_path / f"submission-{label}.json").read_text()) == {
        "attempts": selected
    }


def test_logging_is_idempotent() -> None:
    first = configure_logging()
    second = configure_logging()

    assert first is second
    assert first.name == LOGGER_NAME
    assert len(first.handlers) == 1
