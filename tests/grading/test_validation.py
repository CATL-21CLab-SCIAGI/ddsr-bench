import pytest

from ddsr_bench.grading.validation import (
    extract_answer,
    extract_code,
    validate_code,
)

TEMPLATE = """import sympy as sp

def answer():
    return ...
"""


def test_extracts_one_python_block() -> None:
    response = "Reasoning first.\n```python\nimport sympy as sp\n\ndef answer():\n    return sp.sqrt(2)\n```"

    code = extract_answer(response, TEMPLATE)

    assert code.endswith("return sp.sqrt(2)\n")


def test_uses_first_python_block() -> None:
    response = """```
scratch = 0
```
```python
def answer():
    return 1
```
```python
def answer():
    return 2
```"""

    assert "return 1" in extract_code(response)


def test_accepts_unfenced_code() -> None:
    assert (
        extract_code("def answer():\n    return 1") == "def answer():\n    return 1\n"
    )


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("def answer(:\n    pass", "invalid Python syntax"),
        ("def answer():\n    return 1\ndef helper():\n    return 2", "exactly one"),
        ("import os\ndef answer():\n    return 1", "imports provided"),
        ("def answer():\n    return open('/tests/secret')", "may not use"),
        (
            "import sympy as sp\ndef answer():\n    return sp.__class__",
            "may not access",
        ),
    ],
)
def test_rejects_unsafe_code(code: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_code(code, TEMPLATE)
