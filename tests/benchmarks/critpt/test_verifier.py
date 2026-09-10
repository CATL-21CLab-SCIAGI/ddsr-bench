import json
from pathlib import Path

from ddsr_bench.benchmarks.critpt.verifier import verify

TEMPLATE = """import sympy as sp

p = sp.symbols("p")

def answer(p):
    return ...
"""
REFERENCE = TEMPLATE.replace("def answer", "def real_answer").replace(
    "return ...", "return (p + 1) ** 2"
)


def write(path: Path, name: str, content: str) -> Path:
    target = path / name
    target.write_text(content, encoding="utf-8")
    return target


def write_testcases(path: Path, testcases: list[dict]) -> Path:
    return write(path, "testcases.json", json.dumps(testcases))


def test_reference_verification(tmp_path: Path) -> None:
    answer = TEMPLATE.replace("return ...", "return p**2 + 2*p + 1")

    result = verify(
        write(tmp_path, "answer.py", answer),
        write(tmp_path, "template.py", TEMPLATE),
        write(tmp_path, "reference.py", REFERENCE),
    )

    assert result == {
        "reward": 1.0,
        "mode": "reference",
        "status": "passed",
        "passed": 1,
        "total": 1,
    }


def test_testcases(tmp_path: Path) -> None:
    answer = TEMPLATE.replace("return ...", "return (p + 1) ** 2 if p == 0 else p + 1")
    result = verify(
        write(tmp_path, "answer.py", answer),
        write(tmp_path, "template.py", TEMPLATE),
        write(tmp_path, "reference.py", REFERENCE),
        write_testcases(tmp_path, [{"p": 0}, {"p": 1}]),
    )

    assert result["reward"] == 0
    assert result["status"] == "different"
    assert (result["passed"], result["total"]) == (1, 2)


def test_multiple_parameters(tmp_path: Path) -> None:
    template = "def answer(a, b):\n    return ...\n"
    reference = "def real_answer(a, b):\n    return a + b\n"
    result = verify(
        write(tmp_path, "answer.py", template.replace("...", "a + b")),
        write(tmp_path, "template.py", template),
        write(tmp_path, "reference.py", reference),
        write_testcases(tmp_path, [{"a": 1, "b": 2}]),
    )

    assert result["reward"] == 1


def test_zero_parameters(tmp_path: Path) -> None:
    template = "def answer():\n    return ...\n"
    result = verify(
        write(tmp_path, "answer.py", template.replace("...", "1")),
        write(tmp_path, "template.py", template),
        write(tmp_path, "reference.py", "def real_answer():\n    return 1\n"),
        write_testcases(tmp_path, [{}]),
    )

    assert result["reward"] == 1


def test_testcase_keys(tmp_path: Path) -> None:
    for testcase in ({}, {"p": 1, "extra": 2}):
        result = verify(
            write(tmp_path, "answer.py", TEMPLATE.replace("...", "p")),
            write(tmp_path, "template.py", TEMPLATE),
            write(tmp_path, "reference.py", REFERENCE),
            write_testcases(tmp_path, [testcase]),
        )

        assert result["reward"] == 0
        assert result["error"] == "ValueError"


def test_format_only_verification(tmp_path: Path) -> None:
    answer = TEMPLATE.replace("return ...", "return p + 1")

    result = verify(
        write(tmp_path, "answer.py", answer),
        write(tmp_path, "template.py", TEMPLATE),
    )

    assert result == {"reward": 1.0, "mode": "format", "status": "passed"}


def test_format_only_checks_signature(tmp_path: Path) -> None:
    result = verify(
        write(tmp_path, "answer.py", "def answer():\n    return 1\n"),
        write(tmp_path, "template.py", TEMPLATE),
    )

    assert result["reward"] == 0
    assert result["error"] == "ValueError"


def test_different_answer(tmp_path: Path) -> None:
    answer = TEMPLATE.replace("return ...", "return p + 1")

    result = verify(
        write(tmp_path, "answer.py", answer),
        write(tmp_path, "template.py", TEMPLATE),
        write(tmp_path, "reference.py", REFERENCE),
    )

    assert result["reward"] == 0
    assert result["status"] == "different"


def test_runtime_error(tmp_path: Path) -> None:
    answer = TEMPLATE.replace("return ...", "return 1 / 0")

    result = verify(
        write(tmp_path, "answer.py", answer),
        write(tmp_path, "template.py", TEMPLATE),
        write(tmp_path, "reference.py", REFERENCE),
    )

    assert result["reward"] == 0
    assert result["status"] == "error"
    assert result["error"] == "ZeroDivisionError"
