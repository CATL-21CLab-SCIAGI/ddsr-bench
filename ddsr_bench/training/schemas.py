from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Generation:
    """One model call, or one fixed stage used in its context."""

    id: str
    prompt: tuple[dict[str, str], ...]
    completion: dict[str, str] | None
    source: Literal["model", "fixed"]
    quality: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Canonical, trainer-independent record of one teacher attempt."""

    schema_version: int
    id: str
    benchmark: str
    problem_id: str
    generations: tuple[Generation, ...]
    teacher: dict[str, Any]
    quality: dict[str, Any]
    metadata: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SftSample:
    """One conversational prompt with one supervised assistant completion."""

    id: str
    prompt: tuple[dict[str, str], ...]
    completion: tuple[dict[str, str], ...]
    metadata: dict[str, Any]
