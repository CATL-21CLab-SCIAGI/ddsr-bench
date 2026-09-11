from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module, resources
from pathlib import Path
from typing import Any, Protocol

import yaml

from ddsr_bench.training.schemas import SftSample, Trajectory

from .utils import Resources


class StaticRunner(Protocol):
    async def __call__(
        self, path: Path, *, task_name: str | None = None
    ) -> dict[str, Any]: ...


ResultAdapter = Callable[
    [dict[str, Any], Path],
    tuple[dict[str, Any], Path],
]
Summarizer = Callable[[list[dict[str, Any]]], dict[str, Any]]
TrajectoryAdapter = Callable[
    [Path, dict[str, Any], dict[str, Any], dict[str, Any]],
    Trajectory,
]
SftAdapter = Callable[[Trajectory, str], tuple[SftSample, ...]]
Preparer = Callable[
    [Mapping[str, Any], str | Path | None, str | Path, Resources],
    tuple[Path, ...],
]
Submitter = Callable[[Path, int, str, str, float], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class Benchmark:
    """Capabilities implemented by one benchmark adapter."""

    static_runner: str | None = None
    result_adapter: str | None = None
    summarizer: str | None = None
    trajectory_adapter: str | None = None
    sft_adapter: str | None = None
    preparer: str | None = None
    submitter: str | None = None


BENCHMARKS = {
    "critpt": Benchmark(
        static_runner="ddsr_bench.benchmarks.critpt.evaluation.static:run_job",
        result_adapter="ddsr_bench.benchmarks.critpt.result:trial_fields",
        trajectory_adapter="ddsr_bench.benchmarks.critpt.trajectory:normalize",
        sft_adapter="ddsr_bench.benchmarks.critpt.trajectory:sft_samples",
        preparer="ddsr_bench.benchmarks.critpt.evaluation.prepare:prepare_tasks",
        submitter="ddsr_bench.benchmarks.critpt.evaluation.submit:submit_attempt",
    ),
    "scicode": Benchmark(
        result_adapter="ddsr_bench.benchmarks.scicode.result:trial_fields",
        trajectory_adapter="ddsr_bench.benchmarks.scicode.trajectory:normalize",
        sft_adapter="ddsr_bench.benchmarks.scicode.trajectory:sft_samples",
        preparer="ddsr_bench.benchmarks.scicode.evaluation.prepare:prepare_tasks",
    ),
    "cmphysbench": Benchmark(
        static_runner="ddsr_bench.benchmarks.cmphysbench.evaluation.static:run_job",
        result_adapter="ddsr_bench.benchmarks.cmphysbench.result:trial_fields",
        summarizer="ddsr_bench.benchmarks.cmphysbench.result:summarize",
        trajectory_adapter="ddsr_bench.benchmarks.cmphysbench.trajectory:normalize",
        sft_adapter="ddsr_bench.benchmarks.cmphysbench.trajectory:sft_samples",
        preparer="ddsr_bench.benchmarks.cmphysbench.evaluation.prepare:prepare_tasks",
    ),
}


def benchmark(name: str) -> Benchmark:
    try:
        return BENCHMARKS[name]
    except KeyError as error:
        raise ValueError(f"unknown benchmark {name!r}") from error


def benchmark_config(name: str) -> dict[str, Any]:
    """Load one benchmark's shared configuration."""
    benchmark(name)
    path = resources.files("configs").joinpath("benchmark", f"{name}.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("name") != name:
        raise ValueError(f"invalid benchmark configuration for {name!r}")
    return config


def _resolve(target: str) -> Any:
    module_name, member = target.split(":", maxsplit=1)
    return getattr(import_module(module_name), member)


def static_runner(name: str) -> StaticRunner:
    target = benchmark(name).static_runner
    if target is None:
        raise ValueError(f"benchmark {name!r} has no static runner")
    return _resolve(target)


def result_adapter(name: str) -> ResultAdapter:
    target = benchmark(name).result_adapter
    if target is None:
        raise ValueError(f"benchmark {name!r} has no result adapter")
    return _resolve(target)


def summarizer(name: str) -> Summarizer | None:
    target = benchmark(name).summarizer
    if target is None:
        return None
    return _resolve(target)


def trajectory_adapter(name: str) -> TrajectoryAdapter:
    target = benchmark(name).trajectory_adapter
    if target is None:
        raise ValueError(f"benchmark {name!r} has no trajectory adapter")
    return _resolve(target)


def sft_adapter(name: str) -> SftAdapter:
    target = benchmark(name).sft_adapter
    if target is None:
        raise ValueError(f"benchmark {name!r} has no SFT adapter")
    return _resolve(target)


def preparer(name: str) -> Preparer:
    target = benchmark(name).preparer
    if target is None:
        raise ValueError(f"benchmark {name!r} requires no task preparation")
    return _resolve(target)


def submitter(name: str) -> Submitter:
    target = benchmark(name).submitter
    if target is None:
        raise ValueError(f"benchmark {name!r} does not support submission")
    return _resolve(target)
