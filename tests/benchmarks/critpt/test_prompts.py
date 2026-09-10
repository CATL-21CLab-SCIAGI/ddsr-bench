import pytest

from ddsr_bench.benchmarks.critpt.prompts import parse_prompt, system_prompt


def test_system_styles() -> None:
    one_step = system_prompt("one-step")
    two_step = system_prompt("two-step")

    assert "**Parsing Structure**" in one_step
    assert "**Formatting Compliance**" not in one_step
    assert "**Formatting Compliance**" in two_step
    assert "**Parsing Structure**" not in two_step
    assert "at least 12 significant digits" in one_step


def test_parse_prompt() -> None:
    template = "def answer():\n    return {{value}}"
    rendered = parse_prompt(template)

    assert template in rendered
    assert rendered.startswith("Populate your final answer")
    assert rendered.endswith("\n```")


def test_rejects_unknown_style() -> None:
    with pytest.raises(ValueError, match="unknown prompt style"):
        system_prompt("three-step")  # type: ignore[arg-type]
