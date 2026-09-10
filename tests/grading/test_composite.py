import pytest

from critpt_eval.grading.composite import all_required, weighted


def test_all_required() -> None:
    assert all_required([True, True])
    assert not all_required([True, False])


def test_weighted() -> None:
    assert weighted([True, False, True], [2, 1, 1]) == 0.75


@pytest.mark.parametrize(
    ("checks", "weights", "message"),
    [
        ([], [], "at least one"),
        ([True], [1, 2], "same length"),
        ([True], [-1], "finite"),
        ([True], [0], "positive total"),
    ],
)
def test_invalid_composition(
    checks: list[bool], weights: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        weighted(checks, weights)


def test_checks_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="booleans"):
        all_required([1])  # type: ignore[list-item]
