import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.hipho.data.loader import load_problem
from ddsr_bench.benchmarks.hipho.evaluation.evaluator import _score, evaluate
from ddsr_bench.benchmarks.hipho.evaluation.official.hipho_prompt_inference import (
    SYSTEM_PROMPTS_EN,
    SYSTEM_PROMPTS_ZH,
)
from ddsr_bench.benchmarks.hipho.evaluation.prepare import compile_problem
from ddsr_bench.benchmarks.hipho.evaluation.static import run_trial
from ddsr_bench.benchmarks.hipho.generation.prompts import messages, render
from ddsr_bench.generation.client import ChatResponse, Sampling


def _record() -> dict:
    return {
        "id": "F=MA_2024_01",
        "context": "Given data.",
        "question": "Find the value. [figure1]",
        "marking": [["Award 1.0 pt for the correct result."]],
        "answer": [r"\boxed{2}"],
        "answer_type": ["Numerical Value"],
        "unit": [None],
        "points": [1.0],
        "modality": "text+illustration figure",
        "field": "Mechanics",
        "source": "F=MA_2024",
        "image_question": ["image_question/F=MA_2024_01_1.png"],
    }


def _problem(tmp_path: Path, exam: str = "F=MA_2024"):
    image = tmp_path / "image_question" / "F=MA_2024_01_1.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\x89PNG\r\n\x1a\nbody")
    record = _record() | {"source": exam}
    return load_problem(record, exam, "Use $g=10$.", tmp_path)


class FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    async def chat(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return ChatResponse(self.content, None, "fake", {}, None, 0.1, {}, "stop")

    async def preflight(self) -> None:
        pass


class BrokenCoarseScorer:
    def _evaluate_fine_grained(self, *_):
        return 0.5, [{"score": 0.5}]

    def _evaluate_coarse_grained(self, *_):
        raise AttributeError("malformed extension reference")


def test_public_projection_and_exact_prompt(tmp_path: Path) -> None:
    problem = _problem(tmp_path)
    public = json.dumps(asdict(problem.spec))
    expected = (
        SYSTEM_PROMPTS_EN.replace("{context}", problem.spec.context)
        .replace("{problem}", problem.spec.question)
        .replace("{information}", problem.spec.information)
    )

    assert "Award 1.0" not in public
    assert render(problem.spec) == expected
    assert r"\boxed{{answer}}" in expected
    content = messages(problem.spec)[0]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]


def test_chinese_prompt_parity(tmp_path: Path) -> None:
    problem = _problem(tmp_path, "PanMechanics_2024")
    expected = (
        SYSTEM_PROMPTS_ZH.replace("{context}", problem.spec.context)
        .replace("{problem}", problem.spec.question)
        .replace("{information}", problem.spec.information)
    )

    assert problem.spec.language == "zh"
    assert render(problem.spec) == expected


@pytest.mark.asyncio
async def test_official_answer_level_scoring(tmp_path: Path) -> None:
    problem = _problem(tmp_path)
    judge = FakeClient(r"\boxed{0.0}")

    result = await evaluate(
        r"<answer>[\boxed{2}]</answer>", problem.spec, problem.answer, judge
    )

    assert result["reward"] == 1.0
    assert result["fine_grained_points"] == 0.0
    assert result["coarse_grained_points"] == 1.0
    assert result["judge_calls"] == 1


def test_later_extension_preserves_fine_score(tmp_path: Path) -> None:
    problem = _problem(tmp_path, "CPhO_2025")
    judge = object()

    result = _score(
        BrokenCoarseScorer(), judge, "response", problem.spec, problem.answer
    )

    assert result["fine"] == 0.5
    assert result["coarse"] == 0.0
    assert result["coarse_error"] == "AttributeError"


def test_prepared_task_keeps_reference_private(tmp_path: Path) -> None:
    problem = _problem(tmp_path / "source")
    task = compile_problem(problem, tmp_path / "tasks")

    instruction = (task / "instruction.md").read_text(encoding="utf-8")
    reference = (task / "tests" / "reference.json").read_text(encoding="utf-8")
    assert "Award 1.0" not in instruction
    assert "Award 1.0" in reference


@pytest.mark.asyncio
async def test_static_trial_keeps_reference_private(tmp_path: Path) -> None:
    problem = _problem(tmp_path / "source")
    generation = FakeClient(r"<answer>[\boxed{2}]</answer>")
    judge = FakeClient(r"\boxed{0.0}")

    result = await run_trial(
        problem,
        0,
        tmp_path / "trial",
        generation,
        judge,
        Sampling("teacher", max_tokens=2048, temperature=0.6),
        client_name="vllm",
        judge_name="openai",
        judge_model="gemini-2.5-flash",
        judge_max_tokens=4096,
    )

    response = (tmp_path / "trial" / "agent" / "response.json").read_text(
        encoding="utf-8"
    )
    assert result["status"] == "passed"
    assert "Award 1.0" not in response
