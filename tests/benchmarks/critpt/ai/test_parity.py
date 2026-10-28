"""Optional differential checks against the pinned upstream, without API calls.

Set DDSR_TEST_AUTO_CRITPT to a checkout at UPSTREAM_COMMIT. Only trusted synthetic
answers are executed here; corpus answers must use isolated workers.
"""

import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.ai import grade_response, schemas
from ddsr_bench.benchmarks.critpt.evaluation.ai.answer import _final_answer_code
from ddsr_bench.benchmarks.critpt.evaluation.ai.llm.utils import extract_json_object
from ddsr_bench.benchmarks.critpt.evaluation.ai.rule import (
    categorical,
    numeric,
    symbolic,
)

UPSTREAM_COMMIT = "565d1b0bfab353e17a40a38fe656c3ab0caee660"


@pytest.fixture(scope="module")
def upstream():
    location = os.environ.get("DDSR_TEST_AUTO_CRITPT")
    if not location:
        pytest.skip("set DDSR_TEST_AUTO_CRITPT to the pinned upstream checkout")
    root = Path(location)
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == UPSTREAM_COMMIT
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / "src"))
        yield importlib.import_module("auto_critpt_grader")


def outcome(fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        return (
            result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        )
    except Exception as error:  # noqa: BLE001 - compare both implementations' failures
        return {"exception": type(error).__name__}


@pytest.mark.parametrize(
    "kind,answer,target,options",
    [
        ("floating_numeric", 1.25, 1.25, {"abs_tol": 0.001}),
        ("floating_numeric", 1.3, 1.25, {"rel_tol": 0.001}),
        ("floating_numeric", 1.0, 1, {"abs_tol": 0}),
        ("floating_numeric", 1.0, 1.0, {}),
        ("exact", "sqrt(12)/4", "sqrt(3)/2", {}),
        ("exact", "1", "x", {}),
        ("exact", "1", "fresnels(1)", {}),
        ("symbolic", "x+x", "2*x", {"variables": ["x"]}),
        ("symbolic", "x/10**20", "x/10**20", {"variables": ["x"]}),
        (
            "symbolic",
            "(x-2/7)*(x-5/11)*(x-11/17)",
            "(x-2/7)*(x-5/11)*(x-11/17)",
            {"variables": ["x"]},
        ),
        ("symbolic", "x", "sin(x)**2+cos(x)**2", {}),
        ("categorical", ["A", "B"], ["B", "A"], {}),
        ("categorical", "A", "B", {"choices": ["A"]}),
        ("categorical", "A", "B", {"allowed_aliases": {"A": ["B"], "B": []}}),
    ],
)
def test_rules(upstream, kind, answer, target, options):
    ours = {
        "floating_numeric": numeric.grade_floating_numeric,
        "exact": numeric.grade_exact_numeric,
        "symbolic": symbolic.grade_symbolic,
        "categorical": categorical.grade_categorical,
    }[kind]
    theirs = getattr(upstream, f"grade_{kind}")
    assert outcome(ours, answer, target, **options) == outcome(
        theirs, answer, target, options
    )


@pytest.mark.parametrize("weight", [-1, 0, 1, float("inf"), float("nan")])
def test_plan(upstream, weight):
    fields = {
        "calls": [
            {"primitive": "exact", "target": "", "expected": "2", "weight": weight}
        ]
    }
    for cls in (upstream.GradingPlan, schemas.GradingPlan):
        # Upstream accepts these weights/targets; comparison owns weight handling.
        cls.model_validate(fields)


@pytest.mark.parametrize("weight", [-1, 0, 1])
@pytest.mark.parametrize("passed", [False, True])
def test_composition(upstream, weight, passed):
    from ddsr_bench.benchmarks.critpt.evaluation.ai.rule.compare import compose_grades

    fields = {
        "calls": [
            {
                "primitive": "exact",
                "target": "answer",
                "expected": "2",
                "weight": weight,
            }
        ]
    }
    grade = {
        "primitive": "exact",
        "target": "answer",
        "passed": passed,
        "score": float(passed),
    }
    assert outcome(
        compose_grades, schemas.GradingPlan(**fields), [schemas.PrimitiveGrade(**grade)]
    ) == outcome(
        upstream.compose_grades,
        upstream.GradingPlan(**fields),
        [upstream.PrimitiveGrade(**grade)],
    )


@pytest.mark.parametrize(
    "response",
    [
        "",
        "No function",
        "```python\ndef answer(:\n```",
        "```python\ndef answer():\n return 2.0\n```",
        "```python\nimport os\ndef answer():\n return 2\n```",
        "```python\ndef answer():\n return 1/0\n```",
    ],
)
def test_failure_reports(upstream, response):
    problem = {
        "problem_id": "synthetic",
        "problem_statement": "Return two.",
        "public_solver_output_instructions": "Use answer().",
        "answer_contract": {"answer_type": "exact"},
        "grading_plan": {
            "calls": [{"primitive": "exact", "target": "answer", "expected": "2"}]
        },
    }
    actual = grade_response(problem, response, llm_fallback=False).model_dump(
        mode="json"
    )
    expected = upstream.grade_response(
        problem, response, judge_enabled=False
    ).model_dump(mode="json")
    assert actual == {key: value for key, value in expected.items() if key in actual}


@pytest.mark.parametrize(
    "content",
    [
        '{"verdict":"first","verdict":"last"}',
        'text {bad} then {"verdict":"correct"}',
        '{"verdict":"correct"} {"verdict":"incorrect"}',
        '```json\n{"verdict":"correct",}\n```',
    ],
)
def test_judge_json(upstream, content):
    from auto_critpt_grader.llm_io import extract_json_object as original

    assert outcome(extract_json_object, content) == outcome(original, content)


@pytest.mark.parametrize(
    "response",
    [
        "```python\ndef answer():\n return 1\n```\n```py\ndef answer():\n return 2\n```",
        "def answer(): return 1",
        "```python\nx = 1\n```",
    ],
)
def test_extraction(upstream, response):
    from auto_critpt_grader.primitives import _final_answer_code as original

    assert outcome(_final_answer_code, response) == outcome(original, response)


@pytest.mark.parametrize("composite", [False, True])
def test_reports(upstream, composite):
    calls = [{"primitive": "exact", "target": "answer", "expected": "2"}]
    if composite:
        calls.append({"primitive": "categorical", "target": "choice", "expected": "A"})
    problem = {
        "problem_id": "synthetic",
        "problem_statement": "Return two and A.",
        "public_solver_output_instructions": "Use answer().",
        "answer_contract": {"answer_type": "composite" if composite else "exact"},
        "grading_plan": {"calls": calls},
    }
    response = "```python\ndef answer():\n return {'answer': 2, 'choice': 'A'}\n```"
    ours = grade_response(problem, response, llm_fallback=False).model_dump(mode="json")
    theirs = upstream.grade_response(problem, response, judge_enabled=False).model_dump(
        mode="json"
    )
    # DDSR intentionally omits report version/reward; DDPR owns reward mapping.
    assert ours == {key: value for key, value in theirs.items() if key in ours}


@pytest.mark.parametrize("confidence", [0.89, 0.90, "high"])
def test_equivalence(upstream, confidence):
    from ddsr_bench.benchmarks.critpt.evaluation.ai.llm.equivalence import (
        EquivalenceJudgeConfig,
        LLMEquivalenceJudge,
    )

    payload = json.dumps(
        {
            "verdict": "equivalent",
            "confidence": confidence,
            "rationale": " same ",
            "assumptions_used": [" x real "],
        }
    )
    prompts = []

    def complete(messages, *_):
        prompts.append(messages)
        return payload

    fields = {"primitive": "exact", "target": "answer", "expected": "2"}
    grade = {"primitive": "exact", "target": "answer", "passed": False, "score": 0}
    old = upstream.ExpressionEquivalenceJudge(
        config=upstream.EquivalenceJudgeConfig(model="mock"), completion=complete
    )
    new = LLMEquivalenceJudge(
        config=EquivalenceJudgeConfig(model="mock"), completion=complete
    )
    expected = outcome(
        old.judge,
        call=upstream.GradingCall(**fields),
        actual="2",
        deterministic_grade=upstream.PrimitiveGrade(**grade),
    )
    actual = outcome(
        new.judge,
        call=schemas.GradingCall(**fields),
        answer="2",
        deterministic_grade=schemas.PrimitiveGrade(**grade),
    )
    assert actual == expected
    assert prompts[0] == prompts[1]


@pytest.mark.parametrize("verdict", ["correct", "incorrect", "uncertain", "invalid"])
def test_semantic(upstream, verdict):
    from ddsr_bench.benchmarks.critpt.evaluation.ai.llm.semantic import (
        LLMSemanticJudge,
        SemanticJudgeConfig,
    )

    fields = {
        "problem_id": "synthetic",
        "problem_statement": "Return two.",
        "public_solver_output_instructions": "Explain the result.",
        "answer_type": "exact",
        "reference_answer": "2",
        "reference_solution": "1+1=2",
    }
    prompts = []

    def complete(messages, *_):
        prompts.append(messages)
        return json.dumps(
            {
                "verdict": verdict,
                "confidence": 0.9,
                "solver_answer_summary": " two ",
                "assessment": " correct ",
            }
        )

    old = upstream.LLMSemanticJudge(
        config=upstream.SemanticJudgeConfig(model="mock"), completion=complete
    )
    new = LLMSemanticJudge(
        config=SemanticJudgeConfig(model="mock"), completion=complete
    )
    expected = upstream.grade_response(fields, "two", semantic_judge=old).model_dump(
        mode="json"
    )
    actual = grade_response(fields, "two", semantic_judge=new).model_dump(mode="json")
    assert actual == {key: value for key, value in expected.items() if key in actual}
    middle = len(prompts) // 2
    assert prompts[:middle] == prompts[middle:]


@pytest.mark.parametrize(
    "verdict", ["equivalent", "not_equivalent", "uncertain", "api_error"]
)
@pytest.mark.parametrize("fallback", [False, True])
def test_fallback_reports(upstream, verdict, fallback):
    from ddsr_bench.benchmarks.critpt.evaluation.ai.llm.equivalence import (
        EquivalenceJudgeConfig,
        LLMEquivalenceJudge,
    )

    def complete(*_):
        if verdict == "api_error":
            raise RuntimeError("synthetic provider failure")
        return json.dumps(
            {"verdict": verdict, "confidence": 1, "rationale": "synthetic"}
        )

    old = upstream.ExpressionEquivalenceJudge(
        config=upstream.EquivalenceJudgeConfig(model="mock"), completion=complete
    )
    new = LLMEquivalenceJudge(
        config=EquivalenceJudgeConfig(model="mock"), completion=complete
    )
    problem = {
        "problem_id": "synthetic",
        "problem_statement": "Return x.",
        "public_solver_output_instructions": "Use answer().",
        "answer_contract": {"answer_type": "symbolic"},
        "grading_plan": {
            "calls": [
                {
                    "primitive": "symbolic",
                    "target": "answer",
                    "expected": "x",
                    "args": {"variables": ["x"]},
                }
            ]
        },
    }
    # Unsupported operation stays inert; this never executes fresnels.
    response = (
        "```python\nimport sympy as sp\ndef answer(x):\n return sp.fresnels(x)\n```"
    )
    actual = grade_response(
        problem, response, llm_fallback=fallback, equivalence_judge=new
    ).model_dump(mode="json")
    expected = upstream.grade_response(
        problem, response, judge_enabled=fallback, equivalence_judge=old
    ).model_dump(mode="json")
    assert actual == {key: value for key, value in expected.items() if key in actual}
