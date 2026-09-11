"""Pinned CMPhysBench SEED implementation."""

from dataclasses import dataclass

from ddsr_bench.benchmarks.cmphysbench.data.schemas import AnswerType

from .metric import SEED


@dataclass(frozen=True, slots=True)
class SeedScore:
    points: float
    relative_distance: float
    tree_size: float
    raw_distance: float

    @property
    def reward(self) -> float:
        return self.points / 100

    @property
    def accurate(self) -> bool:
        return abs(self.points - 100) < 1e-4


def score(reference: str, prediction: str, answer_type: AnswerType) -> SeedScore:
    """Apply SEED and expose its diagnostics with named fields."""
    result = SEED(reference, prediction, answer_type)
    return SeedScore(*(float(value) for value in result))


__all__ = ["SeedScore", "score"]
