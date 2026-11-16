import json
from pathlib import Path

from harbor.models.task.config import NetworkMode
from harbor.models.task.task import Task

from ddsr_bench.benchmarks.cmtbench.data.loader import load_problem
from ddsr_bench.benchmarks.cmtbench.evaluation.prepare import (
    compile_problem,
    decode_instruction,
    encode_instruction,
)
from ddsr_bench.benchmarks.utils import Resources

FIXTURE = Path(__file__).parents[2] / "fixtures" / "cmtbench_39.json"


def problem():
    return load_problem(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_compile(tmp_path: Path) -> None:
    source = problem()
    assert source.answer is not None
    task_path = compile_problem(
        source, tmp_path, Resources("ddsr-bench-cmtbench:test", 2, 4096, 120)
    )

    task = Task(task_path)
    assert decode_instruction(encode_instruction(source.spec)) == source.spec
    assert source.answer.solution not in task.instruction
    assert json.loads((task_path / "tests" / "reference.json").read_text()) == {
        "solution": source.answer.solution,
        "parameters": source.answer.parameters,
        "functions": source.answer.functions,
    }
    assert task.config.environment.network_mode == NetworkMode.NO_NETWORK
    assert task.config.environment.docker_image == "ddsr-bench-cmtbench:test"
    assert task.config.artifacts[0].source == "/app/answer.txt"
    assert (task_path / "tests" / "test.sh").stat().st_mode & 0o111
