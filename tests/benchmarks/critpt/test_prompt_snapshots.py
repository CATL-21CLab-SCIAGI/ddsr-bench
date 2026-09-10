from pathlib import Path

import pytest

from ddsr_bench.benchmarks.critpt.generation.prompts import (
    PromptStyle,
    parse_prompt,
    system_prompt,
)

SNAPSHOTS = Path(__file__).parents[2] / "snapshots"


def snapshot(name: str) -> str:
    # Snapshot files have one storage newline that is not part of the prompt.
    return (SNAPSHOTS / f"{name}.txt").read_text(encoding="utf-8")[:-1]


@pytest.mark.parametrize(
    ("style", "name"), [("one-step", "one_step"), ("two-step", "two_step")]
)
def test_system_prompt_matches_upstream(style: PromptStyle, name: str) -> None:
    assert system_prompt(style) == snapshot(name)


def test_parse_prompt_matches_upstream() -> None:
    template = "def answer():\n    return 42"

    assert parse_prompt(template) == snapshot("parse")
