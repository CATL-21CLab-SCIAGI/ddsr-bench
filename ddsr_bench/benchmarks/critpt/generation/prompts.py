from functools import cache
from importlib.resources import files
from typing import Literal

import yaml
from jinja2 import Environment

from ddsr_bench.benchmarks.critpt.data.schemas import (
    AIProblemSpec,
    CritPtSource,
    ProblemSpec,
)

PromptStyle = Literal["one-step", "two-step"]

_ENV = Environment()
_PATHS = {
    CritPtSource.OFFICIAL: files("configs").joinpath(
        "prompts", "critpt", "default.yaml"
    ),
    CritPtSource.AI: files("configs").joinpath("prompts", "critpt", "ai.yaml"),
}


def load_prompts(source: CritPtSource) -> dict:
    """Read source-specific prompt configuration for preparation and generation."""
    return yaml.safe_load(_PATHS[CritPtSource(source)].read_text(encoding="utf-8"))


@cache
def _definitions() -> dict:
    return load_prompts(CritPtSource.OFFICIAL)


def _render(name: str, *, definitions: dict | None = None, **values: object) -> str:
    definitions = _definitions() if definitions is None else definitions
    return _ENV.from_string(definitions[name]).render(values)


def system_prompt(
    style: PromptStyle,
    *,
    source: CritPtSource = CritPtSource.OFFICIAL,
    grader: Literal["rule", "llm"] | None = None,
) -> str:
    """Route system-prompt construction to the source-specific implementation."""
    if style not in ("one-step", "two-step"):
        raise ValueError(f"unknown prompt style: {style}")

    if CritPtSource(source) == CritPtSource.AI:
        return _ai_sys_prompt(style, grader)
    return _official_sys_prompt(style)


def _ai_sys_prompt(style: PromptStyle, grader: Literal["rule", "llm"] | None) -> str:
    """Select solver instructions by how the generated answer will be graded."""
    if style != "one-step":
        raise ValueError("AI answer generation currently requires one-step")
    if grader not in ("rule", "llm"):
        raise ValueError(f"unknown answer grader: {grader}")
    # Read editable AI defaults afresh so resume detects prompt changes.
    definitions = load_prompts(CritPtSource.AI)
    return definitions["SOLVER_SYSTEM_PROMPT"][grader]


def user_prompt(problem: ProblemSpec, style: PromptStyle) -> str:
    """Route user-prompt construction to the source-specific implementation."""
    if CritPtSource(problem.source) == CritPtSource.AI:
        if not isinstance(problem, AIProblemSpec):
            raise TypeError("AI source requires AIProblemSpec")
        return _ai_usr_prompt(problem)
    return _official_usr_prompt(problem, style)


def _official_usr_prompt(problem: ProblemSpec, style: PromptStyle) -> str:
    content = problem.statement
    if style == "one-step" and problem.code_template:
        content = f"{content}\n\n```python\n{problem.code_template}\n```"
    return content


def _ai_usr_prompt(problem: AIProblemSpec) -> str:
    """Reassemble public output instructions using upstream's user envelope."""
    definitions = load_prompts(CritPtSource.AI)
    instructions = problem.answer_instructions
    if not instructions or not instructions.strip():
        instructions = definitions["ANSWER_INSTRUCTIONS"]
    if problem.code_template:
        instructions += f"\n\n```python\n{problem.code_template.rstrip()}\n```"
    return _render(
        "USER_PROMPT",
        definitions=definitions,
        statement=problem.statement,
        label=(
            "Final answer instructions"
            if problem.grader == "rule"
            else "Response instructions"
        ),
        instructions=instructions,
    )


def _official_sys_prompt(style: PromptStyle) -> str:
    """Compose the pinned official CritPt prompt sections."""
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
