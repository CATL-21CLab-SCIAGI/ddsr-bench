from pathlib import Path
from types import SimpleNamespace

import pytest

from critpt_eval.commands.dispatch import LOGGER_NAME, configure_logging, dispatch


def test_collect_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "critpt_eval.commands.dispatch.collect_trials",
        lambda _: {"complete_batches": 2},
    )
    config = SimpleNamespace(
        action="collect", paths=SimpleNamespace(input="harbor-job")
    )

    assert dispatch(config) == "collected 2 complete batches"


def test_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="unknown action"):
        dispatch(SimpleNamespace(action="run"))


def test_export_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "critpt_eval.commands.dispatch.export_trajectories",
        lambda job, output: Path(output) / "trajectories.jsonl",
    )
    monkeypatch.setattr(
        "critpt_eval.commands.dispatch.export_sft",
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


def test_logging_is_idempotent() -> None:
    first = configure_logging()
    second = configure_logging()

    assert first is second
    assert first.name == LOGGER_NAME
    assert len(first.handlers) == 1
