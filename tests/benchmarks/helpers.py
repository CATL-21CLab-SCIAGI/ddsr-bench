from pathlib import Path

from ddsr_bench.benchmarks.collect import collect_trials
from ddsr_bench.benchmarks.registry import sft_adapter
from ddsr_bench.training.export import load_trajectory


def assert_static_pipeline(trial: Path, benchmark: str) -> None:
    """Exercise shared collection and training adapters on one real trial."""
    summary = collect_trials(trial.parent)
    assert summary["problems"] == 1
    assert summary["trials"] == 1
    assert "metrics" in summary

    trajectory = load_trajectory(trial)
    assert trajectory.benchmark == benchmark
    samples = sft_adapter(benchmark)(trajectory, "full")
    assert samples and all(sample.completion for sample in samples)
