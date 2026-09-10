from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields needed to generate an answer for one problem."""

    id: str
    type: Literal["main", "sub"]
    index: int | None
    statement: str
    code_template: str
    source: str
    source_path: Path


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only fields from a CritPt problem record."""

    code: str
    snippet: str
    testcases: tuple[dict[str, Any], ...] | None


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None


@dataclass(frozen=True, slots=True)
class Challenge:
    id: str
    problems: tuple[Problem, ...]
    source_path: Path

    @property
    def main(self) -> Problem:
        return self.problems[0]
