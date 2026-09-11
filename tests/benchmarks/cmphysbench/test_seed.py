import pytest

from ddsr_bench.benchmarks.cmphysbench.data.schemas import AnswerType
from ddsr_bench.benchmarks.cmphysbench.evaluation.seed import score


@pytest.mark.parametrize(
    ("answer_type", "reference", "prediction"),
    [
        ("Expression", r"2 m g + 4\frac{m v_0^2}{l}", r"2 m g + 4\frac{m v_0^2}{l}"),
        ("Equation", r"x^2 + 2x + 1 = 0", r"x^2 + 2x + 1 + 0 = 0"),
        ("Tuple", r"(x, y, z)", r"\left(x, y, z \right)"),
        ("Interval", r"[0, 1]", r"\left[0, 1 \right]"),
        ("Numeric", r"1000 \mathrm{m}", r"1 \mathrm{km}"),
    ],
)
def test_official_examples(
    answer_type: AnswerType, reference: str, prediction: str
) -> None:
    result = score(reference, prediction, answer_type)

    assert result.points == 100
    assert result.reward == 1
    assert result.accurate


def test_partial_score() -> None:
    result = score("x + y", "x + z", "Expression")

    assert 0 < result.points < 100
    assert result.reward == result.points / 100
    assert not result.accurate
