import sympy as sp

from ddsr_bench.benchmarks.cmtbench.evaluation.scorer import equivalent_functional
from ddsr_bench.benchmarks.cmtbench.evaluation.scorer.functional import (
    _simplify_fermions,
)


def test_algebraic_equivalence() -> None:
    assert equivalent_functional(r"\boxed{x+x}", r"\boxed{2x}", "$x$")
    assert not equivalent_functional(r"\boxed{x+1}", r"\boxed{2x}", "$x$")


def test_unordered_expressions() -> None:
    assert equivalent_functional(r"\boxed{y;x}", r"\boxed{x;y}", "$x;y$")
    assert not equivalent_functional(r"\boxed{x}", r"\boxed{x;y}", "$x;y$")


def test_fermionic_rules() -> None:
    annihilator = sp.Symbol("a", commutative=False)
    creator = sp.Symbol("a_dagger", commutative=False)
    operators = {"a": annihilator, "a_dagger": creator}

    # Upstream only rewrites explicit products; SymPy collapses this to Pow.
    assert _simplify_fermions(annihilator**2, operators) == annihilator**2
    assert _simplify_fermions(annihilator * creator, operators) == (
        1 - creator * annihilator
    )
