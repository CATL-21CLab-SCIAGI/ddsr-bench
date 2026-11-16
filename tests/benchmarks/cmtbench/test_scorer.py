import pytest

from ddsr_bench.benchmarks.cmtbench.evaluation.scorer import (
    equivalent,
    extract_solutions,
    parse_numeric,
)


def test_final_box() -> None:
    response = r"Work: \boxed{wrong}. Final: \fbox{\frac{a}{b}; c\quad n>0}."

    assert extract_solutions(response) == (r"\frac{a}{b}", "c")


def test_nested_text_box() -> None:
    assert extract_solutions(r"\boxed{\text{(1, {2})}}") == ("(1, {2})",)


@pytest.mark.parametrize("response", ["", "answer", r"\boxed{}"])
def test_missing_box(response: str) -> None:
    with pytest.raises(ValueError):
        extract_solutions(response)


def test_parse_numeric() -> None:
    parsed = parse_numeric(r"\boxed{x=2; (4.19, -1); 3}")

    assert parsed == (2.0, (4.19, -1.0), 3.0)


def test_numeric_equivalence() -> None:
    assert equivalent((2.000001, (4.0, 1.0)), ((4.0, 1.0), 2.0))
    assert not equivalent((2.00001,), (2.0,))
    assert not equivalent(((1.0, 2.0),), (1.0,))


def test_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="unsupported value"):
        parse_numeric(r"\boxed{'answer'}")
