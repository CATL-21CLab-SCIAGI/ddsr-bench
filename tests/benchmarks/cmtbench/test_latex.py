import pytest

from ddsr_bench.benchmarks.cmtbench.evaluation.scorer import normalize_latex


@pytest.mark.parametrize(
    ("latex", "expression"),
    [
        (r"\frac{a}{b}", "(a)/(b)"),
        (r"\sqrt{x}", "(x)**(1/2)"),
        (r"x^{2}", "x^(2)"),
        (r"\mathrm{sin}(x)", " sin(x)"),
        (r"2 \times \pi", "2 * pi"),
        (r"e^{x}", "E^(x)"),
        (r"\exp{x}", " E^(x)"),
        (r"\left|\frac{x}{2}\right|", " abs((x)/(2))"),
        (r"\Gamma(x)", " gamma(x)"),
        (r"Ai(x)", " airyai(x)"),
        ("√x", "√x"),
    ],
)
def test_upstream_rewrites(latex: str, expression: str) -> None:
    assert normalize_latex(latex) == expression


def test_empty_expression() -> None:
    with pytest.raises(ValueError, match="empty LaTeX"):
        normalize_latex("  ")
