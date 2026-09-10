from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_verifier_image() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text()
    ignored = (ROOT / ".dockerignore").read_text()

    assert dockerfile.startswith("FROM python:3.12-slim\n")
    assert all(name in dockerfile for name in ("numpy", "scipy", "sympy"))
    assert "COPY critpt_eval /opt/critpt-eval/critpt_eval" in dockerfile
    assert "COPY ." not in dockerfile
    assert ignored.splitlines()[0] == "*"
