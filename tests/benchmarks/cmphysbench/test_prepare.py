import json
from pathlib import Path

from harbor.models.task.config import NetworkMode
from harbor.models.task.task import Task

from ddsr_bench.benchmarks.cmphysbench.data.loader import load_problem
from ddsr_bench.benchmarks.cmphysbench.evaluation.prepare import (
    compile_problem,
    decode_instruction,
)
from ddsr_bench.benchmarks.utils import Resources


def test_harbor_task() -> None:
    record = {
        "id": 1,
        "context": "Public context. ",
        "question": "Find the energy.",
        "symbol": "$E$: energy",
        "answer": "SECRET_SOLUTION",
        "final_answer": ["SECRET_REFERENCE"],
        "answer_type": "Expression",
        "topic": "Theory",
    }
    problem = load_problem(record)

    # Public instruction and private reference remain separate.
    instruction = decode_instruction(
        json.dumps(
            {
                "id": "1",
                "context": "Public context. ",
                "question": "Find the energy.",
                "symbols": "$E$: energy",
                "answer_type": "Expression",
                "topic": "Theory",
            }
        )
    )
    assert instruction == problem.spec


def test_compile(tmp_path: Path) -> None:
    problem = load_problem(
        {
            "id": 1,
            "context": "",
            "question": "Find $E$.",
            "symbol": "$E$: energy",
            "answer": "SECRET_SOLUTION",
            "final_answer": ["E"],
            "answer_type": "Expression",
            "topic": "Theory",
        }
    )
    path = compile_problem(
        problem, tmp_path, Resources("ddsr-bench-cmphysbench:test", 2, 4096, 120)
    )

    task = Task(path)
    assert path.name == "1"
    assert "SECRET" not in task.instruction
    assert json.loads((path / "tests" / "reference.json").read_text()) == {
        "answer": "E",
        "answer_type": "Expression",
    }
    assert task.config.environment.network_mode == NetworkMode.NO_NETWORK
    assert task.config.environment.docker_image == "ddsr-bench-cmphysbench:test"
    assert task.config.artifacts[0].source == "/app/answer.txt"
    assert (path / "tests" / "test.sh").stat().st_mode & 0o111
