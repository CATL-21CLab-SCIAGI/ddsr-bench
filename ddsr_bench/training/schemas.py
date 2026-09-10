from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Canonical, trainer-independent record of one teacher attempt."""

    schema_version: int
    id: str
    challenge_id: str | None
    problem_id: str
    problem_type: str | None
    problem_index: int | None
    statement: str
    code_template: str
    messages: tuple[dict[str, str], ...]
    teacher: dict[str, Any]
    quality: dict[str, Any]
    provenance: dict[str, str]


@dataclass(frozen=True, slots=True)
class SftSample:
    """One conversational prompt with one supervised assistant completion."""

    id: str
    prompt: tuple[dict[str, str], ...]
    completion: tuple[dict[str, str], ...]
    metadata: dict[str, Any]
