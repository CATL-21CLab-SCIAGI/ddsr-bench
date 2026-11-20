import tomllib
from pathlib import Path

from ddsr_bench.benchmarks.registry import BENCHMARKS

ROOT = Path(__file__).parents[2]


def test_benchmark_packages_and_configs_are_distributed() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packaged = set(config["tool"]["setuptools"]["packages"])
    patterns = config["tool"]["setuptools"]["package-data"]["configs"]

    for name in BENCHMARKS:
        root = ROOT / "ddsr_bench" / "benchmarks" / name
        expected = {
            ".".join(path.parent.relative_to(ROOT).parts)
            for path in root.rglob("__init__.py")
        }
        assert expected <= packaged
        assert any(pattern.startswith(f"jobs/{name}/") for pattern in patterns)
        assert any(pattern.startswith(f"prompts/{name}/") for pattern in patterns)


def test_critpt_image() -> None:
    dockerfile = (ROOT / "docker" / "critpt" / "Dockerfile").read_text()
    ignored = (ROOT / "docker" / "critpt" / "Dockerfile.dockerignore").read_text()

    assert dockerfile.startswith("FROM python:3.12-slim\n")
    assert all(name in dockerfile for name in ("numpy", "scipy", "sympy"))
    assert "COPY ddsr_bench /opt/ddsr-bench/ddsr_bench" in dockerfile
    assert "COPY ." not in dockerfile
    assert ignored.splitlines()[0] == "*"
    assert "!ddsr_bench/**" in ignored


def test_scicode_image() -> None:
    dockerfile = (ROOT / "docker" / "scicode" / "Dockerfile").read_text()
    ignored = (ROOT / "docker" / "scicode" / "Dockerfile.dockerignore").read_text()

    assert "scicode-bench/SciCode" not in dockerfile
    assert all(name in dockerfile for name in ("h5py", "matplotlib", "numpy", "scipy"))
    assert (
        "COPY datasets/scicode/v2.2/test_data_cleaned.h5 /opt/scicode/test_data.h5"
        in dockerfile
    )
    assert "!datasets/scicode/v2.2/test_data_cleaned.h5" in ignored
    assert "32c5602f2bf08393f917f03ff0b80d40" in dockerfile


def test_cmtbench_image() -> None:
    dockerfile = (ROOT / "docker" / "cmtbench" / "Dockerfile").read_text()

    assert dockerfile.startswith("FROM python:3.12-slim\n")
    assert all(name in dockerfile for name in ("numpy", "regex", "sympy"))
    assert "COPY ddsr_bench /opt/ddsr-bench/ddsr_bench" in dockerfile
    assert "COPY ." not in dockerfile


def test_all_solver_contexts_exclude_local_artifacts():
    from docker.utils.build import exclude_paths

    benchmarks = [name for name, item in BENCHMARKS.items() if item.containerized]
    for benchmark in benchmarks:
        patterns = (
            (ROOT / "docker" / benchmark / "Dockerfile.dockerignore")
            .read_text()
            .splitlines()
        )
        visible = set(exclude_paths(ROOT, patterns))
        assert "ddsr_bench/benchmarks/critpt/evaluation/validation.py" in visible
        assert (
            "ddsr_bench/benchmarks/critpt/evaluation/consensus/execution/worker.py"
            in visible
        )
        assert ".env" not in visible
        assert not any(path.startswith("configs/local/") for path in visible)
