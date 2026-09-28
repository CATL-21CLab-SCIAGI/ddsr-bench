import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from ddsr_bench.benchmarks.collect import collect_trials
from ddsr_bench.benchmarks.critpt.data.schemas import AIProblemSpec, CritPtSource
from ddsr_bench.benchmarks.critpt.evaluation.prepare import encode_instruction
from ddsr_bench.benchmarks.critpt.evaluation.static import run_job
from ddsr_bench.benchmarks.critpt.generation import prompts
from ddsr_bench.benchmarks.critpt.trajectory import sft_samples
from ddsr_bench.generation.client import ChatResponse
from ddsr_bench.training.export import export_sft, export_trajectories, load_trajectory


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [False, True])
@pytest.mark.parametrize("instructions", [None, "", "Use the task instructions."])
async def test_teacher_job(tmp_path: Path, monkeypatch, code, instructions):
    prompt = tmp_path / "ai.yaml"
    definitions = yaml.safe_load(prompts._PATHS[CritPtSource.AI].read_text())
    definitions["SOLVER_SYSTEM_PROMPT"] = {"rule": "Rule role.", "llm": "LLM role."}
    definitions["ANSWER_INSTRUCTIONS"] = "Shared instructions."
    prompt.write_text(yaml.safe_dump(definitions))
    monkeypatch.setitem(prompts._PATHS, CritPtSource.AI, prompt)
    problem = AIProblemSpec(
        id="ai-example",
        type="main",
        index=None,
        statement="Find the result.",
        code_template="def answer():\n    return ...\n" if code else "",
        source=CritPtSource.AI,
        source_path=Path("PRIVATE_SOURCE"),
        answer_instructions=instructions,
        grader="rule" if code else "llm",
    )
    task = tmp_path / "tasks" / problem.id
    task.mkdir(parents=True)
    (task / "instruction.md").write_text(encode_instruction(problem))
    (task / "tests").mkdir()
    (task / "tests/reference.json").write_text('{"snippet": "PRIVATE_ANSWER"}')
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "job_name": "teacher",
                "jobs_dir": str(tmp_path / "harbor"),
                "n_attempts": 1,
                "n_concurrent_trials": 1,
                "agents": [
                    {
                        "name": "ddsr_bench.benchmarks.critpt.evaluation.harbor:CritPtAgent",
                        "model_name": "teacher",
                        "kwargs": {"style": "one-step"},
                    }
                ],
                "datasets": [{"path": str(task.parent)}],
            }
        )
    )
    content = (
        "```python\ndef answer():\n    return 42\n```" if code else "The result is 42."
    )

    class Teacher:
        calls = 0

        async def preflight(self):
            pass

        async def chat(self, messages):
            self.calls += 1
            assert messages[0]["content"] == ("Rule role." if code else "LLM role.")
            assert (instructions or "Shared instructions.") in messages[1]["content"]
            assert (
                "Final answer instructions:" if code else "Response instructions:"
            ) in messages[1]["content"]
            assert ("```python" in messages[1]["content"]) is code
            return ChatResponse(
                content,
                "Teacher reasoning.",
                "teacher",
                {"total_tokens": 20},
                None,
                0.1,
                {},
                "stop",
            )

    teacher = Teacher()
    result = await run_job(config, teacher)
    assert result["validated"] == 1 and teacher.calls == 1
    trial = Path(result["output"]) / "ai-example__attempt-0"
    artifact = trial / "artifacts" / ("answer.py" if code else "answer.txt")
    assert artifact.read_text() == (
        "def answer():\n    return 42\n" if code else content
    )
    record = json.loads((trial / "agent/response.json").read_text())
    assert "PRIVATE" not in json.dumps(record)
    assert record["responses"][0]["reasoning"] == "Teacher reasoning."
    assert record["problem"] == {
        "statement": problem.statement,
        "code_template": problem.code_template,
    }
    assert record["messages"][0]["content"] == ("Rule role." if code else "LLM role.")
    assert json.loads((trial / "validation/result.json").read_text())["reward"] is None
    collection = collect_trials(Path(result["output"]))
    assert collection["batches"][0]["trials"][0]["artifact"].endswith(artifact.name)
    trajectory = load_trajectory(trial)
    assert trajectory.metadata["source"] == CritPtSource.AI
    assert trajectory.metadata["grader"] == problem.grader
    assert trajectory.provenance["prompt_commit"].startswith("565d1b0")
    assert trajectory.quality["verified"] is False
    with pytest.raises(ValueError, match="AI trajectories currently require one-step"):
        sft_samples(
            replace(trajectory, teacher=trajectory.teacher | {"strategy": "two-step"}),
            "answer",
        )
    assert trajectory.generations[0].completion["reasoning"] == "Teacher reasoning."
    exported = export_trajectories(Path(result["output"]), tmp_path / "export")
    assert "PRIVATE" not in exported.read_text()
    for view in ("native", "full", "answer"):
        sample = json.loads(export_sft(exported, tmp_path / view, view).read_text())
        assert sample["prompt"] == [
            {"role": m["role"], "content": m["content"]} for m in record["messages"][:2]
        ]
        assert sample["completion"] == [{"role": "assistant", "content": content}]
        assert sample["metadata"]["quality"]["verified"] is False
    await run_job(config, teacher, resume=True)
    assert teacher.calls == 1
    if instructions:
        (task / "instruction.md").write_text(
            encode_instruction(replace(problem, answer_instructions="Changed."))
        )
    else:
        definitions["ANSWER_INSTRUCTIONS"] = "Changed instructions."
        prompt.write_text(yaml.safe_dump(definitions))
    with pytest.raises(ValueError, match="cannot resume changed"):
        await run_job(config, teacher, resume=True)
    assert teacher.calls == 1
