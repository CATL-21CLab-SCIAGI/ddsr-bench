from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one model request."""

    id: str
    tag: str
    content: str


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Private reference fields published for a gradable problem."""

    value: str
    solution: str


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
