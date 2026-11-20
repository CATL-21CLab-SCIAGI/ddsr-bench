from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Language = Literal["en", "zh"]


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one HiPhO request."""

    id: str
    exam: str
    language: Language
    information: str
    context: str
    question: str
    modality: str
    field: str
    images: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only official solutions and marking scheme."""

    answers: tuple[str, ...]
    answer_types: tuple[str, ...]
    units: tuple[str | None, ...]
    points: tuple[float, ...]
    marking: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
