from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SciCodeStep:
    """One ordered coding step from a SciCode problem."""

    id: str
    statement: str
    function: str
    return_line: str
    background: str
    tests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SciCodeProblem:
    id: str
    dependencies: str
    background: str
    steps: tuple[SciCodeStep, ...]
