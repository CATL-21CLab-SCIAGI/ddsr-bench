import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.olympiadbench.data.loader import load_problem
from ddsr_bench.benchmarks.olympiadbench.evaluation.evaluator import evaluate
from ddsr_bench.benchmarks.olympiadbench.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.olympiadbench.evaluation.scorer import equivalent
from ddsr_bench.benchmarks.olympiadbench.evaluation.static import run_trial
from ddsr_bench.benchmarks.olympiadbench.generation.prompts import messages, user_prompt
from ddsr_bench.generation.client import ChatResponse, Sampling
from tests.benchmarks.helpers import assert_static_pipeline


def _record(kind: str = "OE", modality: str = "TO") -> dict:
    record = {
        "id": 7,
        "question": "Find $x$.",
        "solution": ["secret derivation"],
        "final_answer": ["2"],
        "context": "Given data.",
        "modality": "Text-only" if modality == "TO" else "Multimodal",
        "difficulty": "easy",
        "is_multiple_answer": False,
        "unit": None,
        "answer_type": "Numerical",
        "error": "1e-8",
        "question_type": "Open-ended" if kind == "OE" else "Theorem proof",
        "subfield": "algebra",
        "subject": "Math",
        "language": "English",
    }
    for index in range(1, 10):
        record[f"image_{index}"] = None
    if modality == "MM":
        record["question"] += " <image_1>"
        record["image_1"] = {"bytes": b"\x89PNG\r\n\x1a\nbody", "path": None}
    if kind == "TP":
        record["final_answer"] = None
        record["answer_type"] = None
        record["error"] = None
    return record


def _problem(kind: str = "OE", modality: str = "TO"):
    return load_problem(_record(kind, modality), f"{kind}_{modality}_maths_en_COMP")


class FakeClient:
    async def chat(self, request, **kwargs):
        return ChatResponse(
            r"So the final answer is $\boxed{2}$.",
            None,
            "teacher",
            {},
            None,
            0.1,
            {},
            "stop",
        )

    async def preflight(self) -> None:
        pass


def test_public_projection_and_official_prompt() -> None:
    problem = _problem()
    public = json.dumps(asdict(problem.spec))

    assert "secret derivation" not in public
    assert problem.answer.answers == ("2",)
    assert user_prompt(problem.spec).startswith(
        "The following is an open-ended problem from an International Math competition. "
        "The answer of The problem should be a numerical value."
    )
    assert messages(problem.spec)[0] == {
        "role": "system",
        "content": "You are an AI assistant. Please answer the following Math competition problems as required.",
    }


def test_multimodal_placeholder_becomes_image_part() -> None:
    problem = _problem(modality="MM")
    content = messages(problem.spec)[1]["content"]

    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_official_scorer_behaviors() -> None:
    assert equivalent("42", r"work $\boxed{42}$")
    assert equivalent("50", r"\boxed{0.5}")
    assert equivalent(r"x^2", r"\boxed{x x}")
    assert equivalent(r"[0,\pi]", r"\boxed{[0,3.141592653589793]}")
    assert equivalent("1,2", r"\boxed{2},\boxed{1}")
    assert not equivalent("1,2", r"\boxed{1}")


def test_open_and_proof_evaluation_boundary() -> None:
    open_problem = _problem()
    proof = _problem("TP")

    passed = evaluate(r"\boxed{2}", open_problem.spec, open_problem.answer)
    ungraded = evaluate("a proof", proof.spec, proof.answer)

    assert passed["reward"] == 1.0
    assert passed["status"] == "passed"
    assert ungraded["reward"] is None
    assert ungraded["status"] == "ungraded"


def test_prepared_task_keeps_reference_private(tmp_path: Path) -> None:
    task = compile_problem(_problem(), tmp_path)
    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    reference = (task / "tests" / "reference.json").read_text(encoding="utf-8")

    assert "secret derivation" not in instruction
    assert "secret derivation" in reference


@pytest.mark.asyncio
async def test_static_trial_scores_without_harbor(tmp_path: Path) -> None:
    result = await run_trial(
        _problem(),
        0,
        tmp_path / "trial",
        FakeClient(),
        Sampling("teacher", max_tokens=2048, temperature=0.0),
        client_name="vllm",
    )

    assert result["status"] == "passed"
    response = (tmp_path / "trial" / "agent" / "response.json").read_text(
        encoding="utf-8"
    )
    assert "secret derivation" not in response
    assert_static_pipeline(tmp_path / "trial", "olympiadbench")
