from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AnswerType = Literal["Expression", "Equation", "Tuple", "Interval", "Numeric"]


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one model request."""

    id: str
    context: str
    question: str
    symbols: str
    answer_type: AnswerType
    topic: str


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only official answer."""

    value: str


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
