from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProblemType = Literal["DMRG", "ED", "HF", "Other", "PEPS", "QMC", "SM", "VMC"]


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one model request."""

    id: str
    prompt: str
    type: ProblemType


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only parsing fields and official solution."""

    solution: str
    parameters: str
    functions: str


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
