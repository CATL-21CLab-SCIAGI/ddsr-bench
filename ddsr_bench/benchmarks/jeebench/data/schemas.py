from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ResponseType = Literal["MCQ", "MCQ(multiple)", "Integer", "Numeric"]


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one JEEBench request."""

    id: str
    paper: str
    index: int
    subject: str
    response_type: ResponseType
    question: str


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only official answer."""

    value: str


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
