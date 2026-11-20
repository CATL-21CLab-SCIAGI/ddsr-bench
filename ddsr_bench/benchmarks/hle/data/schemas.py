from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Subset = Literal["Gold", "Revision", "Uncertain"]
AnswerType = Literal["exactMatch", "multipleChoice"]


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one HLE request."""

    id: str
    subset: Subset
    category: str
    subject: str
    question: str
    image: str | None


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only official answer and rationale."""

    answer: str
    type: AnswerType
    rationale: str
    rationale_image: str | None


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
