from pathlib import Path

import pytest

from ddsr_bench.benchmarks import registry
from ddsr_bench.benchmarks.registry import (
    BENCHMARKS,
    dataset_validator,
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
    config = benchmark_config(name)
    assert config["name"] == name
    assert config["edition"]
    assert config["authority"] in ("official", "independent")
    assert config["status"] in (
        "original",
        "revised",
        "corrected",
        "verified",
        "unverified",
    )
    assert len(config["source_sha256"]) == 64
    assert registry.benchmark_config(name)["name"] == name


@pytest.mark.parametrize("name", BENCHMARKS)
def test_benchmark_layout(name: str) -> None:
    root = Path("ddsr_bench/benchmarks") / name
    assert (root / "README.md").is_file()
    assert (root / "PIPELINE.md").is_file()
    assert f"benchmarks/{name}/README.md" in Path("BENCHMARKS.md").read_text()
    assert f"benchmarks/{name}/PIPELINE.md" in Path("PIPELINE.md").read_text()
    if BENCHMARKS[name].containerized:
        assert (Path("docker") / name / "Dockerfile").is_file()
    assert any((Path("configs/jobs") / name).glob("*.yaml"))
    assert any((Path("configs/prompts") / name).iterdir())


def test_unknown_benchmark_config() -> None:
    with pytest.raises(ValueError, match="unknown benchmark"):
        benchmark_config("unknown")
    with pytest.raises(ValueError, match="invalid benchmark name"):
        benchmark_config("../critpt")


@pytest.mark.parametrize("name", BENCHMARKS)
def test_static_capabilities(name: str) -> None:
    target = BENCHMARKS[name].static_runner
    if target is None:
        with pytest.raises(ValueError, match="no static runner"):
            static_runner(name)
    else:
        assert static_runner(name).__name__ == "run_job"


@pytest.mark.parametrize(
    "name", [name for name, value in BENCHMARKS.items() if value.result_adapter]
)
def test_result_capabilities(name: str) -> None:
    assert result_adapter(name).__name__ == "trial_fields"


@pytest.mark.parametrize("name", BENCHMARKS)
def test_summary_capabilities(name: str) -> None:
    summary = summarizer(name)
    if BENCHMARKS[name].summarizer is None:
        assert summary is None
    else:
        assert summary is not None and summary.__name__ == "summarize"


@pytest.mark.parametrize(
    "name", [name for name, value in BENCHMARKS.items() if value.trajectory_adapter]
)
def test_trajectory_capabilities(name: str) -> None:
    assert trajectory_adapter(name).__name__ == "normalize"
    assert sft_adapter(name).__name__ == "sft_samples"


@pytest.mark.parametrize("name", BENCHMARKS)
def test_preparation_capabilities(name: str) -> None:
    assert preparer(name).__name__ == "prepare_tasks"


@pytest.mark.parametrize("name", BENCHMARKS)
def test_dataset_source_capabilities(name: str) -> None:
    assert callable(dataset_validator(name))


def test_submission_capabilities() -> None:
    assert submitter("critpt").__name__ == "submit"
    with pytest.raises(ValueError, match="does not support submission"):
        submitter("cmphysbench")
