import sympy as sp

from critpt_eval.grading.compare import compare


def test_mixed_return_value() -> None:
    x = sp.symbols("x")
    expected = (x**2 + 2 * x + 1, "A", 1.0, 2)
    actual = ((x + 1) ** 2, "a", 1.0 + 5e-13, 2)

    assert compare(actual, expected)


def test_nested_values() -> None:
    assert compare({"numbers": [1.0, 2.0]}, {"numbers": [1.0, 2.0]})
    assert not compare({"numbers": [1.0]}, {"numbers": [1.0, 2.0]})
    assert not compare([], [])


def test_absolute_tolerance_only() -> None:
    assert compare(0.0001, 0.0, rel_tol=0, abs_tol=0.001)
    assert not compare(0.01, 0.0, rel_tol=0, abs_tol=0.001)
    assert compare(sp.Float("1.0001"), sp.Float("1"), rel_tol=0, abs_tol=0.001)


def test_rejects_wrong_return_types() -> None:
    assert not compare([1, 2], (1, 2))
    assert not compare(1, True)
    assert not compare(None, None)
