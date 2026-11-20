from __future__ import annotations

import re
from importlib import resources
from typing import Any

from ddsr_bench.benchmarks.olympiadbench.data.schemas import ProblemSpec
from ddsr_bench.generation.client import ChatMessages

_TYPES = {
    "en": {
        "Numerical": "a numerical value",
        "Expression": "an expression",
        "Equation": "an equation",
        "Interval": "an interval",
    },
    "zh": {
        "Numerical": "数值",
        "Expression": "表达式",
        "Equation": "方程",
        "Interval": "区间",
    },
}


def _template(name: str) -> str:
    return (
        resources.files("configs")
        .joinpath("prompts", "olympiadbench", name)
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )


def system_prompt(problem: ProblemSpec) -> str:
    subject = {"maths": ("Math", "数学"), "physics": ("Physics", "物理")}
    index = 0 if problem.language == "en" else 1
    return _template(f"system-{problem.language}.txt").format(
        subject=subject[problem.subject][index]
    )


def _answer_type(problem: ProblemSpec) -> str:
    types = (problem.answer_type or "").split(",")
    known = _TYPES[problem.language]
    if not types or any(item not in known for item in types):
        return ""
    translated = [known[item] for item in types]
    if problem.language == "en":
        if not problem.multiple_answers:
            return f"The answer of The problem should be {translated[0]}. "
        if len(set(translated)) == 1:
            return f"The problem has multiple answers, each of them should be {translated[0]}. "
        return (
            "The problem has multiple answers, with the answers in order being "
            + ", ".join(translated)
            + ". "
        )
    if not problem.multiple_answers:
        return f"，答案类型为{translated[0]}"
    if len(set(translated)) == 1:
        return f"，题目有多个答案，答案类型均为{translated[0]}"
    return "，题目有多个答案，答案类型分别为" + "、".join(translated)


def user_prompt(problem: ProblemSpec) -> str:
    subject = {
        "maths": ("Math", "数学"),
        "physics": ("Physics", "物理"),
    }[
        problem.subject
    ][0 if problem.language == "en" else 1]
    if problem.kind == "TP":
        prefix = _template(f"proof-{problem.language}.txt").format(subject=subject)
    else:
        multiple = bool(problem.multiple_answers)
        boxed = (
            r"\boxed{multiple answers connected with commas}"
            if multiple and problem.language == "en"
            else (
                r"\boxed{用英文逗号连接的多个答案}"
                if multiple
                else r"\boxed{answer}" if problem.language == "en" else r"\boxed{答案}"
            )
        )
        unit = ""
        if problem.unit:
            boxed += f"({problem.unit})"
            unit = (
                r", note that the unit of the answer should not be included in \boxed{}"
                if problem.language == "en"
                else r"，注意答案的单位不要放在\boxed{}中"
            )
        prefix = _template(f"open-{problem.language}.txt").format(
            subject=subject,
            answer_type=_answer_type(problem),
            boxed=boxed,
            unit=unit,
        )
    question = problem.question
    if problem.subject == "physics" and problem.context:
        question = f"{problem.context}\n{question}"
    return f"{prefix}\n\n{question}"


def _content(problem: ProblemSpec) -> str | list[dict[str, Any]]:
    prompt = user_prompt(problem)
    if not problem.images:
        return prompt
    parts: list[dict[str, Any]] = []
    for piece in re.split(r"(<image_\d+>)", prompt):
        match = re.fullmatch(r"<image_(\d+)>", piece)
        if match is None:
            if piece:
                parts.append({"type": "text", "text": piece})
            continue
        index = int(match.group(1))
        parts.append(
            {"type": "image_url", "image_url": {"url": problem.images[index - 1]}}
        )
    return parts


def messages(problem: ProblemSpec) -> ChatMessages:
    """Render the pinned official system/user conversation, including images."""
    return (
        {"role": "system", "content": system_prompt(problem)},
        {"role": "user", "content": _content(problem)},
    )
