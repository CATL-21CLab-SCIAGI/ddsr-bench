from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Example:
    question: str
    choices: tuple[str, str, str, str]
    answer: str


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one MMLU request."""

    id: str
    subject: str
    category: str
    question: str
    choices: tuple[str, str, str, str]
    examples: tuple[Example, ...]


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only answer letter."""

    letter: str


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
