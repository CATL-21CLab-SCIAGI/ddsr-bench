import json
from pathlib import Path

import pytest
from harbor.models.task.task import Task

from ddsr_bench.benchmarks.critpt.data.ai import load_problem, load_problems
from ddsr_bench.benchmarks.critpt.data.schemas import (
    AIProblemSpec,
    AnswerSpec,
    CritPtSource,
)
from ddsr_bench.benchmarks.critpt.evaluation.prepare import (
    decode_instruction,
    encode_instruction,
    prepare_tasks,
)
from ddsr_bench.benchmarks.utils import Resources


@pytest.mark.parametrize("indexed", [False, True])
def test_preparation(tmp_path: Path, indexed: bool) -> None:
    record = {
        "problem_id": "same-original-id",
        "problem_statement": "Find the coefficient.",
        "public_solver_output_instructions": (
            "Return the result.\n```python\ndef answer():\n    return ...\n```\nUse a float."
        ),
        "reference_solution": "SECRET_SOLUTION\n```python\nx = 0\n```\n"
        "```python\ndef answer():\n    return 1.447\n```",
        "reasoning_plan": {"checkpoints": ["SECRET_STEP"]},
        "source_ids": ["SECRET_SOURCE"],
    }
    record.update(
        {
            "answer_contract": {"answer_type": "exact"},
            "final_answer": {"answer": 1.447},
            "grading_plan": {"calls": []},
        }
    )
    source = tmp_path / "input"
    for i in range(2):
        folder = source / str(i)
        folder.mkdir(parents=True)
        (folder / "problem.json").write_text(
            json.dumps(record | {"problem_statement": f"Question {i}"})
        )
    (source / "INDEX.json").write_text("{}")
    if indexed:
        (source / "CORPUS_INDEX.json").write_text("{}")
    problems = load_problems(source)
    assert len({p.spec.id for p in problems}) == 2
    assert all(isinstance(p.answer, AnswerSpec) for p in problems)
    assert all(p.answer.testcases is None for p in problems)
    tasks = prepare_tasks(
        {"name": "critpt", "source": "critpt-ai"},
        source,
        tmp_path / "tasks",
        Resources("test", 1, 1024, 120),
    )
    for path in tasks:
        task = Task(path)
        assert task.config.metadata == {}
        assert "[metadata]" not in (path / "task.toml").read_text()
        private = json.loads((path / "tests/reference.json").read_text())
        original = source / private["source_path"]
        assert original.is_file()
        assert private["problem_id"] == json.loads(original.read_text())["problem_id"]
        assert not Path(private["source_path"]).is_absolute()
        single = prepare_tasks(
            {"name": "critpt", "source": "critpt-ai"},
            original,
            tmp_path / "single" / path.name,
            Resources("test", 1, 1024, 120),
        )[0]
        assert json.loads((single / "tests/reference.json").read_text())[
            "source_path"
        ] == (private["source_path"] if indexed else "problem.json")
        public = json.loads(task.instruction)
        assert set(public) == {
            "id",
            "type",
            "index",
            "statement",
            "code_template",
            "answer_instructions",
            "grader",
            "source",
        }
        assert "SECRET" not in task.instruction
        assert "source_path" not in public and "problem_id" not in public
        assert public["answer_instructions"] == "Return the result.\n\nUse a float."
        assert public["grader"] == "rule"
        assert public["type"] == "main" and public["index"] is None
        private.pop("source_path")
        snippet = private.pop("snippet")
        assert private.pop("reference_check") == "matched"
        assert snippet == record["final_answer"]["answer"]
        assert type(snippet) is float
        assert private.pop("code") == (path / "tests/reference.py").read_text()
        assert private == {
            k: v
            for k, v in record.items()
            if k
            not in (
                "problem_statement",
                "public_solver_output_instructions",
                "final_answer",
            )
        }
        assert not (path / "tests/template.py").exists()
        assert public["code_template"] == "def answer():\n    return ...\n"
        assert "exit 2" in (path / "tests/test.sh").read_text()
        decoded = decode_instruction(task.instruction)
        assert isinstance(decoded, AIProblemSpec)
        assert decoded.source is CritPtSource.AI
        assert decoded.answer_instructions == public["answer_instructions"]


@pytest.mark.parametrize(
    "body,expected,status",
    [
        ("return 2", 3, "different"),
        ("return 3*sp.sqrt(3)/8", "3*sqrt(3)/8", "matched"),
        ("x = 3\n    return x", 3, "unresolved"),
        ("return 1 + 2", "3", "unresolved"),
        ("return 3", "3", "matched"),
        ("return 3\n\nx = 1", 3, "unresolved"),
        ("return 3\n\ndef answer():\n    return 3", 3, "unresolved"),
        ('return {"K"}', ["K"], "matched"),
        ("return 1.447", 1.447, "matched"),
        ("return (", "3", "unresolved"),
        ("return", "3", "unresolved"),
        ("return 1", ["K"], "unresolved"),
        ('return {"K": 1}', ["K"], "unresolved"),
        ('return [1, "K"]', ["K"], "unresolved"),
        ('return {"K"}', [1, "K"], "unresolved"),
        ("return {[1]}", ["K"], "unresolved"),
    ],
)
def test_reference_check(tmp_path, body, expected, status):
    record = {
        "problem_id": "example",
        "problem_statement": "Find a value.",
        "public_solver_output_instructions": "```python\nimport sympy as sp\ndef answer():\n    return ...\n```",
        "reference_solution": f"```python\nimport sympy as sp\ndef answer():\n    {body}\n```",
        "reasoning_plan": {},
        "source_ids": [],
        "answer_contract": {
            "answer_type": "categorical" if isinstance(expected, list) else "exact"
        },
        "final_answer": {"answer": expected},
        "grading_plan": {},
    }
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(record))
    problem = load_problem(path)
    assert problem.answer.snippet == expected
    assert type(problem.answer.snippet) is type(expected)
    assert problem.answer.reference_check == status
    assert problem.spec.answer_instructions == ""
    (task,) = prepare_tasks(
        {"name": "critpt", "source": "critpt-ai"},
        path,
        tmp_path / "tasks",
        Resources("test", 1, 1024, 120),
    )
    private = json.loads((task / "tests/reference.json").read_text())
    assert private["snippet"] == expected and type(private["snippet"]) is type(expected)
    assert private["reference_check"] == status
    assert "reference_check" not in json.loads((task / "instruction.md").read_text())
    record["final_answer"]["extra"] = "must not be lost"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="must contain only 'answer'"):
        load_problem(path)
    record.pop("grading_plan")
    path.write_text(json.dumps(record))
    with pytest.raises(TypeError, match="expected grading_plan"):
        load_problem(path)


def test_llm_based(tmp_path: Path, monkeypatch) -> None:
    record = {
        "problem_id": "llm-based-example",
        "problem_statement": "Find the exponent relation.",
        "public_solver_output_instructions": (
            "Give a clear, self-contained solution in natural language. Include enough "
            "reasoning to make the result evaluable and state each requested answer "
            "item explicitly. Mathematical notation and LaTeX are welcome."
        ),
        "reference_answer": "SECRET: β + γ = 2",
        "reference_solution": "SECRET_SOLUTION",
        "reasoning_plan": {"checkpoints": ["SECRET_REASONING"]},
        "source_ids": ["SECRET_SOURCE"],
        "answer_type": "exact",
        "answer_items": [{"item_id": "answer_1", "target": "SECRET_TARGET"}],
        "evidence_ids": ["SECRET_EVIDENCE"],
    }
    source = tmp_path / "problem.json"
    source.write_text(json.dumps(record))
    problem = load_problem(source)
    assert isinstance(problem.answer, AnswerSpec)
    assert problem.spec.answer_instructions is None
    assert problem.spec.code_template == problem.answer.code == ""
    (task,) = prepare_tasks(
        {"name": "critpt", "source": "critpt-ai"},
        source,
        tmp_path / "tasks",
        Resources("test", 1, 1024, 120),
    )
    public = Task(task).instruction
    assert "SECRET" not in public
    assert json.loads(public)["answer_instructions"] is None
    assert json.loads(public)["grader"] == "llm"
    decoded = decode_instruction(public)
    assert isinstance(decoded, AIProblemSpec) and decoded.answer_instructions is None
    private = json.loads((task / "tests/reference.json").read_text())
    assert "reference_check" not in private
    assert private.pop("snippet") == record["reference_answer"]
    assert private.pop("source_path") == "problem.json"
    assert private.pop("code") == ""
    assert private == {
        k: v
        for k, v in record.items()
        if k
        not in (
            "problem_statement",
            "public_solver_output_instructions",
            "reference_answer",
        )
    }
    assert not (task / "tests/reference.py").exists()
    assert not (task / "tests/template.py").exists()
    record["public_solver_output_instructions"] = "Different instructions."
    source.write_text(json.dumps(record))
    problem = load_problem(source)
    assert problem.spec.answer_instructions == "Different instructions."
    decoded = decode_instruction(encode_instruction(problem.spec))
    assert decoded.answer_instructions == "Different instructions."
    # One source-to-config mapping serves both the loader and prompt rendering.
    from ddsr_bench.benchmarks.critpt.generation import prompts

    assert prompts.user_prompt(decoded, "one-step").endswith("Different instructions.")
    config = tmp_path / "ai.yaml"
    config.write_text(
        "ANSWER_INSTRUCTIONS: Different instructions.\n"
        "USER_PROMPT: '{{ instructions }}'\n"
    )
    monkeypatch.setitem(prompts._PATHS, CritPtSource.AI, config)
    problem = load_problem(source)
    assert problem.spec.answer_instructions is None
    assert prompts.user_prompt(problem.spec, "one-step") == "Different instructions."
    config.write_text(
        config.read_text().replace("Different instructions.", "New default.")
    )
    problem = load_problem(source)
    assert problem.spec.answer_instructions == "Different instructions."
    assert prompts.user_prompt(problem.spec, "one-step") == "Different instructions."
