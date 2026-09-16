import json
from pathlib import Path

from harbor.models.task.config import NetworkMode
from harbor.models.task.task import Task

from ddsr_bench.benchmarks.phybench.data.loader import load_problem
from ddsr_bench.benchmarks.phybench.evaluation.prepare import (
    compile_problem,
    decode_instruction,
    encode_instruction,
)
from ddsr_bench.benchmarks.utils import Resources

FIXTURE = Path(__file__).parents[2] / "fixtures" / "phybench_133.json"


def problem():
    return load_problem(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_instruction() -> None:
    source = problem()

    restored = decode_instruction(encode_instruction(source.spec))

    assert restored == source.spec
    assert "natural coordinate system" not in encode_instruction(source.spec)


def test_compile(tmp_path: Path) -> None:
    source = problem()
    path = compile_problem(
        source, tmp_path, Resources("ddsr-bench-phybench:test", 2, 4096, 120)
    )

    task = Task(path)
    assert path.name == "133"
    assert "natural coordinate system" not in task.instruction
    assert json.loads((path / "tests" / "reference.json").read_text()) == {
        "answer": source.answer.value
    }
    assert task.config.environment.network_mode == NetworkMode.NO_NETWORK
    assert task.config.environment.docker_image == "ddsr-bench-phybench:test"
    assert task.config.artifacts[0].source == "/app/answer.txt"
    assert (path / "tests" / "test.sh").stat().st_mode & 0o111
