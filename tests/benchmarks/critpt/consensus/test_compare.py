import pytest
import sympy as s

from ddsr_bench.benchmarks.critpt.evaluation.consensus.compare import Comparator
from ddsr_bench.benchmarks.critpt.evaluation.consensus.policy import MODES, SKIP
from ddsr_bench.benchmarks.critpt.evaluation.consensus.wire import decode, encode


def test_scope_is_disjoint_and_complete():
    active = [n for v in MODES.values() for n in v]
    assert len(active) == len(set(active)) == 61
    assert set(active) | SKIP == set(range(1, 71))
    assert [
        len(MODES[x])
        for x in ("value", "symbolic", "set", "domain", "series", "function")
    ] == [31, 12, 6, 6, 4, 2]


def test_tiny_values_and_arbitrary_precision_integers():
    c = Comparator(57, [])
    assert c.compare(decode(encode(1e-27)), decode(encode(2e-27))).status == "different"
    assert c.compare(s.S.Zero, decode(encode(1e-27))).status == "different"
    c = Comparator(50, [])
    assert c.compare(s.Integer(10**300), s.Integer(10**300 + 1)).status == "different"


def test_precision_depends_on_field_not_reference_python_type():
    c = Comparator(44, [])
    a = decode(encode((4, 1, -9.80028)))
    b = decode(encode((4, 1, -9.8)))
    assert c.compare(a, b).status == "matched"
    assert c.compare(decode(encode((4.001, 1, -9.8))), b).status == "different"
    c = Comparator(31, [])
    assert (
        c.compare(decode(encode((800.4, 2.01))), decode(encode((800.0, 2.0)))).status
        == "matched"
    )
    assert (
        c.compare(decode(encode((800.0, 2.03))), decode(encode((800.0, 2.0)))).status
        == "different"
    )


def test_set_bijection_cannot_reuse_one_reference_element():
    c = Comparator(4, [])
    assert (
        c.compare(
            frozenset({(s.Integer(1), s.Integer(-1)), (s.Integer(2), s.Integer(1))}),
            frozenset({(s.Integer(2), s.Integer(1)), (s.Integer(1), s.Integer(-1))}),
        ).status
        == "matched"
    )
    c = Comparator(58, [])
    assert c.compare(frozenset({"4.70"}), frozenset({"4.7"})).status == "different"


def test_root_set_equivalence_without_positive_k_assumption():
    c = Comparator(40, ["chi", "kappa", "sigma", "k"])
    chi, kappa, sigma, k = [c.canonical[n] for n in ["chi", "kappa", "sigma", "k"]]
    a = frozenset(
        {
            (
                -s.I * sigma * k**6
                + sign * k**3 * s.sqrt(4 * chi * kappa - sigma**2 * k**6)
            )
            / (2 * chi)
            for sign in [-1, 1]
        }
    )
    b = frozenset(
        {
            (
                -s.I * sigma * k**6
                + sign * s.sqrt(4 * chi * kappa * k**6 - sigma**2 * k**12)
            )
            / (2 * chi)
            for sign in [-1, 1]
        }
    )
    assert c.compare(a, b).status == "matched"
    assert c.compare(frozenset({x + 1 for x in a}), b).status == "different"


def test_continuation_accepts_removable_singularity_but_checks_explicit_endpoint():
    c = Comparator(5, ["alpha"])
    x = c.canonical["alpha"]
    formula = -1 - (1 - x) * s.log(1 - x) / x
    good = s.Piecewise((0, s.Eq(x, 0)), (-1, s.Eq(x, 1)), (formula, True))
    wrong = s.Piecewise((2, s.Eq(x, 0)), (-1, s.Eq(x, 1)), (formula, True))
    assert c.compare(formula, good).status == "matched"
    assert c.compare(wrong, good).status == "different"


def test_complex_identity_and_nonreal_counterexample():
    c = Comparator(39, ["n", "np", "g", "gamma", "alpha"])
    a = c.canonical["alpha"]
    assert c.compare(a * s.conjugate(a), s.Abs(a) ** 2).status == "matched"
    assert c.compare(a**2, s.Abs(a) ** 2).status == "different"


def test_interval_openness_and_symbolic_k_are_required():
    c = Comparator(62, ["phi", "k", "k_value"], 1)
    k = c.canonical["k"]
    a = s.Interval.open(0, s.acos((k - 2) / (k - 1)))
    b = s.Interval.open(0, 2 * s.asin(1 / s.sqrt(2 * k - 2)))
    assert c.compare(a, b).status == "matched"
    assert c.compare(s.Interval(0, a.end), b).status == "different"
    assert c.compare(s.Interval.open(0, s.pi / 2), b).status == "different"


def test_series_truncation_and_lower_order_mutation():
    c = Comparator(66, ["q"])
    q = c.canonical["q"]
    base = 1 + 2 * q + 3 * q**15
    assert c.compare(base + 100 * q**16, base).status == "matched"
    assert c.compare(base + q**14, base).status == "different"
    assert c.compare(base + 1 / q, base).status == "unknown"
    c = Comparator(59, ["M", "epsilon", "a"])
    M, e, a = [c.canonical[n] for n in ["M", "epsilon", "a"]]
    assert (
        c.compare(
            frozenset({(M - 1, e / a + e**2)}), frozenset({(M - 1, e / a)})
        ).status
        == "matched"
    )
    assert (
        c.compare(frozenset({(M - 1, 2 * e / a)}), frozenset({(M - 1, e / a)})).status
        == "different"
    )


def test_asymptotic_ignores_constants_but_preserves_log_coefficients():
    c = Comparator(49, ["z", "K"])
    z, K = [c.canonical[n] for n in ["z", "K"]]
    base = z**2 + K * z * s.log(z) + s.log(z) ** 2
    assert c.compare(base + 5 + K + 1 / z, base).status == "matched"
    assert c.compare(base + s.log(z), base).status == "different"


def test_formal_trace_is_not_replaced_by_scalar_samples():
    c = Comparator(65, ["psi", "tr"])
    p = c.canonical["psi"]
    tr = s.Function("tr")
    assert c.compare(frozenset({tr(p * p)}), frozenset({tr(p**2)})).status == "matched"
    assert (
        c.compare(frozenset({tr(p) ** 2}), frozenset({tr(p**2)})).status == "different"
    )


def test_wire_rejects_code_in_numeric_or_function_fields():
    with pytest.raises(ValueError):
        decode({"t": "decimal", "v": "__import__('os').system('echo BAD')"})
    with pytest.raises(KeyError):
        decode({"t": "expr", "f": "eval", "v": []})
    assert decode(
        encode(s.Piecewise((1, s.Symbol("x") > 0), (0, True)))
    ) == s.Piecewise((1, s.Symbol("x") > 0), (0, True))


def test_approximate_set_matching_uses_bijection_not_greedy_pairing():
    c = Comparator(1, [])
    a = frozenset({(s.Float("1"), s.Float("1")), (s.Float("1"), s.Float("1.015"))})
    b = frozenset({(s.Float("1"), s.Float("1.005")), (s.Float("1.005"), s.Float("1"))})
    assert c.compare(a, b).status == "matched"


def test_piecewise_regions_and_middle_continuation():
    c = Comparator(24, ["y", "p_z", "epsilon_IR", "mu"])
    y = c.canonical["y"]
    a = s.atan(s.sqrt(1 - 2 * y) / y) / s.sqrt(1 - 2 * y)
    filled = s.Piecewise((2, s.Eq(y, s.Rational(1, 2))), (a, True))
    assert c.compare(a, filled, (1,)).status == "matched"
    assert c.compare(a + 1, filled, (1,)).status == "different"


def test_sampled_mismatch_reports_an_actual_counterexample():
    c = Comparator(39, ["n", "np", "g", "gamma", "alpha"])
    a = c.canonical["alpha"]
    result = c.compare(a**2, s.Abs(a) ** 2)
    assert result.status == "different"
    assert result.details["point"]["alpha"]
    assert result.details["actual"] != result.details["expected"]
