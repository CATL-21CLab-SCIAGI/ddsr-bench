"""Pinned PHYBench EED implementation."""

from dataclasses import dataclass

from .metric import EED


@dataclass(frozen=True, slots=True)
class EedScore:
    points: float
    relative_distance: float
    tree_size: float
    raw_distance: float

    @property
    def reward(self) -> float:
        return self.points / 100

    @property
    def accurate(self) -> bool:
        return self.points == 100


def score(reference: str, prediction: str) -> EedScore:
    """Apply EED and expose its diagnostics with named fields."""
    result = EED(reference, prediction)
    return EedScore(*(float(value) for value in result))


__all__ = ["EedScore", "score"]
