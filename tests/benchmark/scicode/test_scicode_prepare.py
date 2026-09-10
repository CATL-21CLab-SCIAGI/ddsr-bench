import json
from pathlib import Path

from harbor.models.task.config import NetworkMode
from harbor.models.task.task import Task

from critpt_eval.benchmark.prepare import Resources
from critpt_eval.benchmark.scicode.loader import load_problem
from critpt_eval.benchmark.scicode.prepare import compile_problem

FIXTURE = Path(__file__).parents[2] / "fixtures" / "scicode_19.json"


def test_compiles_scicode_problem(tmp_path: Path) -> None:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problem = load_problem(record)
    resources = Resources("critpt-eval-scicode:test", 2, 4096, 1800)

    path = compile_problem(problem, tmp_path, resources)

    task = Task(path)
    instruction = json.loads(task.instruction)
    assert path.name == "19"
    assert instruction["id"] == "19"
    assert [step["id"] for step in instruction["steps"]] == ["19.1", "19.2"]
    assert "tests" not in instruction["steps"][0]
    assert "assert np.allclose" not in task.instruction

    cases = json.loads((path / "tests" / "cases.json").read_text())
    assert cases[0]["id"] == "19.1"
    assert cases[0]["tests"] == record["sub_steps"][0]["test_cases"]
    assert task.config.environment.network_mode == NetworkMode.NO_NETWORK
    assert task.config.environment.docker_image == "critpt-eval-scicode:test"
    assert task.config.artifacts[0].source == "/app/solution.py"
    assert (path / "tests" / "test.sh").stat().st_mode & 0o111
