from pathlib import Path

import pytest

from ddsr_bench.benchmarks import registry
from ddsr_bench.benchmarks.registry import (
    BENCHMARKS,
    preparer,
    result_adapter,
    sft_adapter,
    static_runner,
    submitter,
    summarizer,
    trajectory_adapter,
)
from ddsr_bench.benchmarks.utils import benchmark_config


@pytest.mark.parametrize("name", BENCHMARKS)
def test_benchmark_configs(name: str) -> None:
    assert benchmark_config(name)["name"] == name
    assert registry.benchmark_config(name)["name"] == name


@pytest.mark.parametrize("name", BENCHMARKS)
def test_benchmark_layout(name: str) -> None:
    root = Path("ddsr_bench/benchmarks") / name
    assert (root / "README.md").is_file()
    assert (root / "PIPELINE.md").is_file()
    assert (Path("docker") / name / "Dockerfile").is_file()
    assert any((Path("configs/jobs") / name).glob("*.yaml"))
    assert any((Path("configs/prompts") / name).iterdir())


def test_unknown_benchmark_config() -> None:
    with pytest.raises(ValueError, match="unknown benchmark"):
        benchmark_config("unknown")
    with pytest.raises(ValueError, match="invalid benchmark name"):
        benchmark_config("../critpt")


def test_static_capabilities() -> None:
    assert static_runner("critpt").__name__ == "run_job"
    assert static_runner("cmphysbench").__name__ == "run_job"
    assert static_runner("phybench").__name__ == "run_job"
    assert static_runner("cmtbench").__name__ == "run_job"
    assert static_runner("prismphysics").__name__ == "run_job"
    with pytest.raises(ValueError, match="no static runner"):
        static_runner("scicode")


@pytest.mark.parametrize(
    "name", [name for name, value in BENCHMARKS.items() if value.result_adapter]
)
def test_result_capabilities(name: str) -> None:
    assert result_adapter(name).__name__ == "trial_fields"


def test_summary_capabilities() -> None:
    assert summarizer("cmphysbench").__name__ == "summarize"
    assert summarizer("phybench").__name__ == "summarize"
    assert summarizer("cmtbench").__name__ == "summarize"
    assert summarizer("prismphysics").__name__ == "summarize"
    assert summarizer("critpt") is None


@pytest.mark.parametrize(
    "name", [name for name, value in BENCHMARKS.items() if value.trajectory_adapter]
)
def test_trajectory_capabilities(name: str) -> None:
    assert trajectory_adapter(name).__name__ == "normalize"
    assert sft_adapter(name).__name__ == "sft_samples"


def test_preparation_capabilities() -> None:
    assert preparer("critpt").__name__ == "prepare_tasks"
    assert preparer("scicode").__name__ == "prepare_tasks"
    assert preparer("cmphysbench").__name__ == "prepare_tasks"
    assert preparer("phybench").__name__ == "prepare_tasks"
    assert preparer("cmtbench").__name__ == "prepare_tasks"
    assert preparer("prismphysics").__name__ == "prepare_tasks"


def test_submission_capabilities() -> None:
    assert submitter("critpt").__name__ == "submit"
    with pytest.raises(ValueError, match="does not support submission"):
        submitter("cmphysbench")
