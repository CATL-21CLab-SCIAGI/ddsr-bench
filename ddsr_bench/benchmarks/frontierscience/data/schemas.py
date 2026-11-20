from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Track = Literal["olympiad", "research"]
Subject = Literal["biology", "chemistry", "physics"]


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used for one FrontierScience request."""

    id: str
    task_group_id: str
    track: Track
    subject: Subject
    prompt: str


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only short answer or ten-point rubric."""

    reference: str


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
