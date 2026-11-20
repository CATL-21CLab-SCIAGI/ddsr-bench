from importlib import import_module
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.utils import Resources, benchmark_config


@pytest.mark.parametrize(
    ("name", "loader"),
    [
        ("cmphysbench", "load_file"),
        ("cmtbench", "load_file"),
        ("phybench", "load_file"),
        ("prismphysics", "load_directory"),
        ("scicode", "load_file"),
        ("physics", "load_directory"),
        ("hle", "load_directory"),
    ],
)
def test_prepares_local_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    loader: str,
) -> None:
    module = import_module(f"ddsr_bench.benchmarks.{name}.evaluation.prepare")
    source = tmp_path / "source"
    received = []

    def load(*args):
        received.append(args)
        return []

    monkeypatch.setattr(module, loader, load)
    if name == "hle":
        monkeypatch.setattr(module, "select", lambda problems: problems)
    tasks = module.prepare_tasks(
        benchmark_config(name),
        source,
        tmp_path / "tasks",
        Resources(f"ddsr-bench-{name}:test", 2, 4096, 120),
    )

    assert tasks == ()
    assert received[0][0] == source
