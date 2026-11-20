from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Setting = Literal["standard10", "standard4", "vision"]
Mode = Literal["cot", "direct"]


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one MMMU-Pro request."""

    id: str
    raw_id: str
    setting: Setting
    subject: str
    domain: str
    question: str
    choices: tuple[str, ...]
    images: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only answer and released explanation."""

    letter: str
    explanation: str


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
