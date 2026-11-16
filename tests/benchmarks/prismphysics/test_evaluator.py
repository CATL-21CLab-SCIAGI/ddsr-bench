import json
from pathlib import Path

from ddsr_bench.benchmarks.prismphysics.data.loader import THRESHOLD, load_problem
from ddsr_bench.benchmarks.prismphysics.evaluation import evaluator
from ddsr_bench.benchmarks.prismphysics.evaluation.evaluator import (
    evaluate,
    extract_formulas,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "prismphysics_02_3006.json"


def answer():
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return load_problem(record, "02").answer


def test_extraction() -> None:
    response = r"Text $$x^2$$ then $$x = 2$$ and $$y \approx 3.0$$."

    assert extract_formulas(response) == ("x = 2", "y = 3.0")


def test_ancestor_closure() -> None:
    reference = answer()
    assert reference is not None

    result = evaluate(r"$$C=\frac{\varepsilon_0 L^2}{d}$$", reference, THRESHOLD)

    assert result.score == 1
    assert result.correct is True
    assert result.direct_matches == (2,)
    assert result.credited == (1, 2)
    assert result.matches[0].answer_index == 0


def test_partial_credit() -> None:
    reference = answer()
    assert reference is not None

    result = evaluate(r"$$C=\varepsilon S/d$$", reference, THRESHOLD)

    assert result.score == 0.5
    assert result.correct is False
    assert result.direct_matches == (1,)
    assert result.credited == (1,)


def test_no_relational_formula() -> None:
    reference = answer()
    assert reference is not None

    result = evaluate(r"The result is $$42$$.", reference, THRESHOLD)

    assert result.score == 0
    assert result.answer_formula_count == 0
    assert result.direct_matches == ()


def test_invalid_pair_does_not_abort(monkeypatch) -> None:
    reference = answer()
    assert reference is not None

    def compare(expected, candidate, **kwargs):
        if candidate.startswith("bad"):
            raise ValueError("unparsable")
        return (expected.replace(" ", "") == candidate.replace(" ", ""), "test")

    monkeypatch.setattr(evaluator, "whether_rel_latex_correct_with_units", compare)
    result = evaluate(
        r"$$bad = input$$ $$C = \frac{\varepsilon S}{d}$$",
        reference,
        THRESHOLD,
    )

    assert result.direct_matches == (1,)
