from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal


class CritPtSource(StrEnum):
    OFFICIAL = "critpt-official"
    AI = "critpt-ai"

    @classmethod
    def _missing_(cls, value: object) -> CritPtSource | None:
        # Older prepared official tasks used the benchmark name as their source.
        if value == "critpt":
            return cls.OFFICIAL
        return None


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields needed to generate an answer for one problem."""

    id: str
    type: Literal["main", "sub"]
    index: int | None
    statement: str
    code_template: str
    source: CritPtSource
    source_path: Path
    # How answers are graded; None means unspecified, not ungraded.
    grader: Literal["rule", "llm"] | None = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class AnswerSpec[Snippet]:
    """Verifier-only fields; snippet preserves the source's final-answer format."""

    code: str
    snippet: Snippet
    testcases: tuple[dict[str, Any], ...] | None


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec[str] | None


@dataclass(frozen=True, slots=True)
class Challenge:
    id: str
    problems: tuple[Problem, ...]
    source_path: Path

    @property
    def main(self) -> Problem:
        return self.problems[0]


@dataclass(frozen=True, slots=True)
class AIProblemSpec(ProblemSpec):
    """AI public fields; answers graded by an LLM currently need no code template."""

    # Problem-specific one-stage guidance; empty/None selects the prompt default.
    # Two-stage generation must adapt this guidance rather than reuse it verbatim.
    answer_instructions: str | None


@dataclass(frozen=True, slots=True)
class AIAnswerSpec(AnswerSpec[str | float | list[Any]]):
    """Native final answer in snippet; answers graded by an LLM currently have no code."""

    # Original corpus ID for provenance; may repeat across source records.
    # ProblemSpec.id is the distinct, content-hash-based prepared task ID.
    problem_id: str
    reference_solution: str
    reasoning_plan: dict[str, Any]
    source_ids: list[str]
    answer_contract: dict[str, Any] | None = None
    grading_plan: dict[str, Any] | None = None
    answer_type: str | None = None
    answer_items: list[dict[str, Any]] | None = None
    evidence_ids: list[str] | None = None
    # Static code-to-answer agreement, not a correctness grade; absent for LLM grading.
    reference_check: Literal["matched", "different", "unresolved"] | None = None


@dataclass(frozen=True, slots=True)
class AIProblem:
    spec: AIProblemSpec
    answer: AIAnswerSpec
