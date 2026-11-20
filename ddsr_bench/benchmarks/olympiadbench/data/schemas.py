from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["OE", "TP"]
Modality = Literal["MM", "TO"]
Subject = Literal["maths", "physics"]
Language = Literal["en", "zh"]
Source = Literal["CEE", "COMP"]


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields required by the official prompt and model request."""

    id: str
    raw_id: int
    subset: str
    kind: Kind
    modality: Modality
    subject: Subject
    language: Language
    source: Source
    subfield: str
    context: str | None
    question: str
    images: tuple[str, ...]
    answer_type: str | None
    multiple_answers: bool | None
    unit: str | None


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only official solution, final answer, and tolerance."""

    solutions: tuple[str, ...]
    answers: tuple[str, ...] | None
    tolerance: str | None


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
