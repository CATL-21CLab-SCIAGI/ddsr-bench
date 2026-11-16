import pytest

from ddsr_bench.benchmarks.cmtbench.evaluation.scorer import (
    canonicalize,
    parse_functions,
    parse_parameters,
)


@pytest.mark.parametrize(
    ("latex", "name", "introduced"),
    [
        (r"c_s^\dagger", "c_s_dagger", {"c_s_dagger"}),
        (r"a_{i,j}^\prime", "a_i_j_prime", {"a_i_j_prime"}),
        (r"x^{2}", "x^{2}", set()),
        (r"\epsilon_i", r"\epsilon_i", {r"\epsilon_i"}),
    ],
)
def test_canonical_names(latex: str, name: str, introduced: set[str]) -> None:
    assert canonicalize(latex) == (name, frozenset(introduced))


def test_parameters() -> None:
    parsed = parse_parameters(r"$k;N; (c_s^\dagger, NC); (c_s, NC)$")

    assert set(parsed) == {"k", "N", "c_s_dagger", "c_s"}
    assert parsed["k"].is_commutative
    assert not parsed["c_s_dagger"].is_commutative
    assert not parsed["c_s"].is_commutative


def test_function_indices() -> None:
    parsed = parse_functions(r"(c_s^\dagger, NC); (c_s, NC), (s,\uparrow,\downarrow)")

    assert set(parsed) == {
        "c_uparrow_dagger",
        "c_downarrow_dagger",
        "c_uparrow",
        "c_downarrow",
    }
    assert all(not function.is_commutative for function in parsed.values())
