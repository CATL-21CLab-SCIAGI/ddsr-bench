from ddsr_bench.benchmarks.prismphysics.evaluation.gradephyx import (
    simplify_latex_expr,
    whether_rel_latex_correct_with_units,
)
from ddsr_bench.benchmarks.prismphysics.evaluation.gradephyx.formula import (
    format_units_latex,
)


def test_pinned_upstream_parity_cases() -> None:
    pairs = (
        (r"C=\frac{\varepsilon S}{d}", r"C=\varepsilon S/d"),
        (r"v=3\unit{km/s}", r"v=3000\unit{m/s}"),
        (r"E=mc^2", r"m=E/c^2"),
    )

    assert all(whether_rel_latex_correct_with_units(*pair)[0] for pair in pairs)
    assert not whether_rel_latex_correct_with_units(r"x=1", r"x=2")[0]


def test_repository_normalizer_compatibility() -> None:
    assert simplify_latex_expr(r"\dot{x}_0 = v") == r"\dot x_0 = v"
    assert simplify_latex_expr(r"V^*=E") == r"V^\notion_times=E"
    assert simplify_latex_expr(r"\int_{\infty}^{R} f(r)dr=0") == (
        r"\int_\infty^{R} f(r)dr=0"
    )


def test_numeric_unit_component() -> None:
    assert format_units_latex("0.001*km/s") == r"0.001*\km/s"
