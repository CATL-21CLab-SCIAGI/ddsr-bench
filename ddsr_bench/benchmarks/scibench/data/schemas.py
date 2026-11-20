from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Split = Literal["text", "image"]
Strategy = Literal["zero", "zero-cot", "few", "few-cot"]


@dataclass(frozen=True, slots=True)
class Example:
    """One public solved example used by the official few-shot prompt."""

    question: str
    unit: str
    answer: str
    solution: str


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one SciBench request."""

    id: str
    split: Split
    textbook: str
    problem_id: str
    question: str
    unit: str
    image: str | None
    examples: tuple[Example, ...]


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only released numeric answer and original unit."""

    number: str
    unit: str


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
