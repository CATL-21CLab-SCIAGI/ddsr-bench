from decimal import Decimal

import numpy as np
import pytest
import sympy as sp

from ddsr_bench.grading.numeric import exact, floating


def test_exact_numbers() -> None:
    assert exact(2, 2)
    assert exact(Decimal("0.1"), Decimal("0.1"))
    assert exact(np.int64(2), sp.Integer(2))
    assert not exact(2**60, 2**60 + 1)
    assert not exact(Decimal("0.1"), 0.1)


@pytest.mark.parametrize("value", [True, "1", None, float("nan"), float("inf")])
def test_exact_rejects_nonfinite_or_non_numeric(value: object) -> None:
    assert not exact(value, value)


def test_floating_tolerances() -> None:
    assert floating(1.0001, 1.0, rel_tol=0, abs_tol=0.001)
    assert floating(1_000_001, 1_000_000, rel_tol=1e-6, abs_tol=0)
    assert not floating(1.01, 1.0, rel_tol=0, abs_tol=0.001)
    assert floating(sp.Rational(1, 3), 1 / 3, rel_tol=1e-12, abs_tol=0)
    assert not floating(2**60, 2**60 + 1, rel_tol=0, abs_tol=0)


def test_floating_rejects_invalid_values() -> None:
    assert not floating(float("nan"), 1, rel_tol=0, abs_tol=0)
    assert not floating("1.0", 1, rel_tol=0, abs_tol=0)
    with pytest.raises(ValueError, match="negative"):
        floating(1, 1, rel_tol=-1, abs_tol=0)
    with pytest.raises(TypeError, match="finite real"):
        floating(1, 1, rel_tol=True, abs_tol=0)
