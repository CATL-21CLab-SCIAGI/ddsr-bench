from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one GPQA request."""

    id: str
    cohort: str
    domain: str
    subdomain: str
    question: str
    choices: tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only randomized answer letter and official explanation."""

    letter: str
    explanation: str


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
