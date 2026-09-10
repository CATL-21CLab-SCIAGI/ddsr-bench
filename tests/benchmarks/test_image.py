from pathlib import Path

ROOT = Path(__file__).parents[2]


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
    assert "COPY datasets/scicode/test_data.h5 /opt/scicode/test_data.h5" in dockerfile
    assert "!datasets/scicode/test_data.h5" in ignored
