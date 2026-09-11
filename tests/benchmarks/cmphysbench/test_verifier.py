import pytest

from ddsr_bench.benchmarks.cmphysbench.evaluation.verifier import (
    extract_boxed,
    verify,
)


def test_nested_box() -> None:
    response = r"Earlier \boxed{wrong}; final \boxed{\frac{x^{2}}{y}}."

    assert extract_boxed(response) == "wrong"


@pytest.mark.parametrize(
    "response",
    ["no final answer", r"\boxed{unfinished"],
)
def test_missing_box(response: str) -> None:
    assert extract_boxed(response) == ""
    assert verify(response, "x", "Expression")["error"] == "MissingBox"


def test_exact_score() -> None:
    result = verify(
        r"reasoning \boxed{\frac{x^{2}}{y}}", r"\frac{x^2}{y}", "Expression"
    )

    assert result["seed_score"] == 100
    assert result["reward"] == 1
    assert result["status"] == "passed"
    assert result["prediction"] == r"\frac{x^{2}}{y}"


def test_partial_score() -> None:
    result = verify(r"\boxed{x + z}", "x + y", "Expression")

    assert 0 < result["reward"] < 1
    assert result["status"] == "different"


def test_timeout() -> None:
    result = verify(r"\boxed{x}", "x", "Expression", timeout_sec=0)

    assert result["reward"] == 0
    assert result["status"] == "timeout"
    assert result["error"] == "TimeoutError"
