import json
from pathlib import Path

from ddsr_bench.benchmarks.phybench.evaluation.verifier import (
    extract_boxed,
    normalize_reference,
    verify,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "phybench_133.json"


def test_extracts_final_box() -> None:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    answer = record["answer"].removeprefix("$$").removesuffix("$$")
    response = (
        r"An intermediate result is \boxed{x}. "
        f"Therefore the answer is \\boxed{{{answer}}}."
    )

    assert extract_boxed(response) == answer


def test_nested_and_escaped_braces() -> None:
    response = r"Work outside the box. \boxed{\frac{1}{2} + \{x\}} trailing text"

    assert extract_boxed(response) == r"\frac{1}{2} + \{x\}"


def test_invalid_box() -> None:
    assert extract_boxed("no final answer") == ""
    assert extract_boxed(r"\boxed{") == ""
    assert extract_boxed(r"\boxed{}") == ""


def test_reference_wrappers() -> None:
    assert normalize_reference(r"$$ x^2 $$") == "x^2"
    assert normalize_reference(r"\[ x^2 \]") == "x^2"
    assert normalize_reference(r"$ x^2 $") == "x^2"
    assert normalize_reference(r"$$\boxed{x^2}$$") == "x^2"


def test_real_answer() -> None:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = verify(
        r"Derivation followed by \boxed{\frac{v^3}{R v_x}}.",
        record["answer"],
    )

    assert result["status"] == "passed"
    assert result["eed_score"] == 100
    assert result["reward"] == 1
    assert "answer" not in result


def test_failures() -> None:
    assert verify("no box", "x")["error"] == "MissingBox"
    assert verify(r"\boxed{x}", "")["error"] == "MissingReference"
    assert verify(r"\boxed{x}", "x", timeout_sec=0)["status"] == "timeout"
