import json
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.phybench.evaluation.eed import EedScore, score

FIXTURE = Path(__file__).parents[2] / "fixtures" / "phybench_133.json"


def test_official_examples() -> None:
    reference = r"2 m g + 4\frac{mv_0^2}{l}"

    exact = score(reference, r"2 m g+4\frac{mv_0^2}{l}")
    partial = score(reference, r"2 m g+2\frac{mv_0^2}{l}")

    assert exact.points == 100
    assert exact.accurate
    assert partial.points == pytest.approx(46.6666666667)
    assert partial.relative_distance == pytest.approx(2 / 15)
    assert partial.tree_size == 15
    assert partial.raw_distance == 2


def test_real_reference() -> None:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))

    result = score(record["answer"], r"\frac{v^3}{R v_x}")

    assert result.points == 100
    assert result.reward == 1


def test_invalid_prediction() -> None:
    result = score("x", "")

    assert result.points == 0
    assert not result.accurate


def test_accuracy_requires_full_score() -> None:
    result = EedScore(99.99999, 0.0, 1.0, 0.0)

    assert not result.accurate
