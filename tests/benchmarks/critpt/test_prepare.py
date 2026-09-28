import json
from dataclasses import replace
from pathlib import Path

import pytest
from harbor.models.task.config import NetworkMode
from harbor.models.task.task import Task

from ddsr_bench.benchmarks.critpt.data.loader import load_challenge
from ddsr_bench.benchmarks.critpt.data.schemas import CritPtSource, ProblemSpec
from ddsr_bench.benchmarks.critpt.evaluation.prepare import (
    compile_challenge,
    compile_problem,
    decode_instruction,
    encode_instruction,
)
from ddsr_bench.benchmarks.utils import Resources

FIXTURE = Path(__file__).parents[2] / "fixtures" / "quantum_error_correction.json"


def test_compiles_harbor_tasks(tmp_path: Path) -> None:
    challenge = load_challenge(FIXTURE)
    resources = Resources("ddsr-bench:test", 2, 4096, 120)

    paths = compile_challenge(challenge, tmp_path, resources)

    assert len(paths) == 4
    task = Task(paths[1])
    instruction = json.loads(task.instruction)
    assert instruction["id"] == "quantum_error_correction_sub_0"
    assert instruction["type"] == "sub"
    assert instruction["index"] == 0
    assert instruction["source"] == "critpt-official"
    decoded = decode_instruction(task.instruction)
    assert type(decoded) is ProblemSpec
    assert decoded.source is CritPtSource.OFFICIAL
    assert "answer_code" not in task.instruction
    assert "answer_only_code" not in task.instruction

    config = task.config
    assert config.environment.docker_image == "ddsr-bench:test"
    assert config.environment.network_mode == NetworkMode.NO_NETWORK
    assert config.environment.cpus == 2
    assert config.environment.memory_mb == 4096
    assert config.agent.timeout_sec == 120
    assert config.verifier.timeout_sec == 120
    assert config.verifier.network_mode == NetworkMode.NO_NETWORK
    assert config.artifacts[0].source == "/app/answer.py"
    assert (paths[1] / "tests" / "template.py").exists()
    assert "def real_answer" in (paths[1] / "tests" / "reference.py").read_text()
    assert (paths[1] / "tests" / "test.sh").stat().st_mode & 0o111


def test_compiles_answer_testcases(tmp_path: Path) -> None:
    problem = load_challenge(FIXTURE).main
    assert problem.answer is not None
    answer = replace(problem.answer, testcases=({"p": 0.1},))
    problem = replace(problem, answer=answer)

    task = compile_problem(
        problem, tmp_path, Resources("ddsr-bench:test", 2, 4096, 120)
    )

    testcases = json.loads((task / "tests" / "testcases.json").read_text())
    assert testcases == [{"p": 0.1}]


@pytest.mark.parametrize("source", ["unknown", None])
def test_unknown_source(source):
    with pytest.raises(ValueError, match="CritPtSource"):
        decode_instruction(json.dumps({"source": source}))


@pytest.mark.parametrize("source", ["critpt", "critpt-official"])
@pytest.mark.parametrize("legacy", [False, True])
def test_official_source_alias(source, legacy):
    problem = load_challenge(FIXTURE).main.spec
    data = json.loads(encode_instruction(problem))
    data["source"] = source
    assert data["grader"] is None
    if legacy:
        del data["grader"]  # Older official tasks omitted this optional field.
    decoded = decode_instruction(json.dumps(data))
    assert type(decoded) is ProblemSpec
    assert decoded.source is CritPtSource.OFFICIAL
    assert decoded.grader is None
    assert json.loads(encode_instruction(decoded))["source"] == "critpt-official"
