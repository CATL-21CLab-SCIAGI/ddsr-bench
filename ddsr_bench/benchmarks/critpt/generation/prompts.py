from functools import cache
from importlib.resources import files
from typing import Literal

import yaml
from jinja2 import Environment

PromptStyle = Literal["one-step", "two-step"]

_ENV = Environment()
_PATH = files("configs").joinpath("prompts", "critpt", "default.yaml")


@cache
def _definitions() -> dict:
    return yaml.safe_load(_PATH.read_text(encoding="utf-8"))


def _render(name: str, **values: object) -> str:
    return _ENV.from_string(_definitions()[name]).render(values)


def system_prompt(style: PromptStyle) -> str:
    """Render CritPt's one-step or two-step system prompt."""
    if style not in ("one-step", "two-step"):
        raise ValueError(f"unknown prompt style: {style}")

    definitions = _definitions()
    precision = _render(
        "SYSTEM_PROMPT_INSTR_PRECISION",
        PROMPT_SPECS_PRECISION_DECIMAL=definitions["PROMPT_SPECS_PRECISION_DECIMAL"],
    )
    final_answer = _render(
        "SYSTEM_PROMPT_INSTR_FA", SYSTEM_PROMPT_INSTR_PRECISION=precision
    )
    return _render(
        "SYSTEM_PROMPT",
        PROMPT_SPECS_STYLE=style,
        SYSTEM_PROMPT_INSTR_DERIVATION=_render("SYSTEM_PROMPT_INSTR_DERIVATION"),
        SYSTEM_PROMPT_INSTR_MATH_FORMAT=_render("SYSTEM_PROMPT_INSTR_MATH_FORMAT"),
        SYSTEM_PROMPT_INSTR_CONVENTION=_render("SYSTEM_PROMPT_INSTR_CONVENTION"),
        SYSTEM_PROMPT_INSTR_FA=final_answer,
        SYSTEM_PROMPT_INSTR_RESULT_FORMAT=_render("SYSTEM_PROMPT_INSTR_RESULT_FORMAT"),
        SYSTEM_PROMPT_INSTR_RESULT_FORMAT_CODE=_render(
            "SYSTEM_PROMPT_INSTR_RESULT_FORMAT_CODE"
        ),
    )


def parse_prompt(code_template: str) -> str:
    """Render CritPt's second-turn formatting prompt."""
    return _render("PARSE_PROMPT", PROMPT_SPECS_ANSWER_CODE_TEMPLATE=code_template)
