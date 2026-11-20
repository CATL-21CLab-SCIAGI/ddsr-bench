import json
from pathlib import Path

from ddsr_bench.benchmarks.hle.data.loader import load_problem
from ddsr_bench.benchmarks.hle.evaluation.prepare import (
    compile_problem,
    decode_instruction,
    decode_reference,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "hle_gold.json"


def test_prepared_task_separates_reference(tmp_path: Path) -> None:
    problem = load_problem(json.loads(FIXTURE.read_text(encoding="utf-8")))
    task = compile_problem(problem, tmp_path)
    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    reference = (task / "tests" / "reference.json").read_text(encoding="utf-8")

    assert decode_instruction(instruction) == problem.spec
    assert decode_reference(reference) == problem.answer
    assert '"answer"' not in instruction
    assert "rationale" not in instruction
    assert '"answer": "D"' in reference


def test_empty_released_rationale_is_valid() -> None:
    reference = json.dumps(
        {
            "answer": "42",
            "type": "exactMatch",
            "rationale": "",
            "rationale_image": None,
        }
    )

    assert decode_reference(reference).rationale == ""
