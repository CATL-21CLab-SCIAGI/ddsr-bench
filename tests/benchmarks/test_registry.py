import pytest

from ddsr_bench.benchmarks.registry import (
    BENCHMARKS,
    benchmark_config,
    preparer,
    result_adapter,
    sft_adapter,
    static_runner,
    submitter,
    summarizer,
    trajectory_adapter,
)


@pytest.mark.parametrize("name", BENCHMARKS)
def test_benchmark_configs(name: str) -> None:
    assert benchmark_config(name)["name"] == name


def test_static_capabilities() -> None:
    assert static_runner("critpt").__name__ == "run_job"
    assert static_runner("cmphysbench").__name__ == "run_job"
    with pytest.raises(ValueError, match="no static runner"):
        static_runner("scicode")


@pytest.mark.parametrize("name", BENCHMARKS)
def test_result_capabilities(name: str) -> None:
    assert result_adapter(name).__name__ == "trial_fields"


def test_summary_capabilities() -> None:
    assert summarizer("cmphysbench").__name__ == "summarize"
    assert summarizer("critpt") is None


@pytest.mark.parametrize("name", BENCHMARKS)
def test_trajectory_capabilities(name: str) -> None:
    assert trajectory_adapter(name).__name__ == "normalize"
    assert sft_adapter(name).__name__ == "sft_samples"


def test_preparation_capabilities() -> None:
    assert preparer("critpt").__name__ == "prepare_tasks"
    assert preparer("scicode").__name__ == "prepare_tasks"
    assert preparer("cmphysbench").__name__ == "prepare_tasks"


def test_submission_capabilities() -> None:
    assert submitter("critpt").__name__ == "submit_attempt"
    with pytest.raises(ValueError, match="does not support submission"):
        submitter("cmphysbench")
