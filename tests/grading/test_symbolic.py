from pathlib import Path

import sympy as sp

from ddsr_bench.benchmarks.critpt.data.loader import load_challenge
from ddsr_bench.grading.symbolic import symbolic

FIXTURE = Path(__file__).parent.parent / "fixtures" / "quantum_error_correction.json"


def test_critpt_reference_expression() -> None:
    reference = load_challenge(FIXTURE).main.answer
    assert reference is not None
    namespace: dict = {}
    exec(reference.code, namespace)  # noqa: S102 - trusted upstream fixture
    p = namespace["p"]
    expected = namespace["real_answer"](p)

    assert symbolic(sp.cancel(expected), expected)


def test_equivalent_expressions() -> None:
    x = sp.symbols("x")

    assert symbolic((x + 1) ** 2, x**2 + 2 * x + 1)
    assert symbolic(sp.sin(x) ** 2 + sp.cos(x) ** 2, 1)
    assert not symbolic(x + 1, x + 2)


def test_rejects_invalid_expressions() -> None:
    x = sp.symbols("x")

    assert not symbolic("x + 1", x + 1)
    assert not symbolic(True, 1)
    assert not symbolic(sp.zoo, sp.zoo)
    assert not symbolic([x], [x])
