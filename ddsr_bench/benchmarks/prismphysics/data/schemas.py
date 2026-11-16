from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Subquestion:
    letter: str
    prompt: str


@dataclass(frozen=True, slots=True)
class Image:
    caption: str
    location: str


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Public fields used to construct one text-only model request."""

    id: str
    raw_id: int
    shard: str
    domain: str
    context: str
    source: str
    subquestions: tuple[Subquestion, ...]
    images: tuple[Image, ...]


@dataclass(frozen=True, slots=True)
class FormulaNode:
    index: int
    formula: str
    dependencies: tuple[int, ...]
    final: bool = False


@dataclass(frozen=True, slots=True)
class AnswerSpec:
    """Verifier-only formula DAG; reference solutions are not retained."""

    formulas: tuple[FormulaNode, ...]


@dataclass(frozen=True, slots=True)
class Problem:
    spec: ProblemSpec
    answer: AnswerSpec | None
