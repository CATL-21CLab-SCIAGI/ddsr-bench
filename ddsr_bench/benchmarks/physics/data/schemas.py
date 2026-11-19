from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Split = Literal["validation", "test"]
AnswerType = Literal["numeric", "symbolic", "text"]
Domain = Literal[
    "Atomic Physics",
    "Classical Mechanics",
    "Electromagnetism",
    "Optics",
    "Quantum Mechanics",
    "Thermodynamics and Statistical Mechanics",
]


@dataclass(frozen=True, slots=True)
class Figure:
    """One public image supplied to a multimodal model."""

    url: str


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one PHYSICS request."""

    id: str
    split: Split
    domain: Domain
    question: str
    figures: tuple[Figure, ...]


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only reference solution and independently scored answers."""

    solution: str
    final_answers: tuple[str, ...]
    labels: tuple[str, ...]
    types: tuple[AnswerType, ...]


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
