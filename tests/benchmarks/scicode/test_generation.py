from hashlib import sha256

import pytest

from ddsr_bench.benchmarks.scicode.data.loader import load_fixed
from ddsr_bench.benchmarks.scicode.data.schemas import SciCodeProblem, SciCodeStep
from ddsr_bench.benchmarks.scicode.generation.runner import (
    _template,
    extract_code,
    generate,
    render_prompt,
)
from ddsr_bench.generation.client import ChatResponse


def problem() -> SciCodeProblem:
    def step(number: int) -> SciCodeStep:
        return SciCodeStep(
            f"7.{number}",
            f"Solve step {number}.",
            f"def step_{number}():",
            "    return value",
            f"Background {number}.",
            (),
        )

    return SciCodeProblem("7", "import numpy as np", "Background", (step(1), step(2)))


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    async def preflight(self) -> None:
        pass

    async def chat(self, messages: tuple[dict[str, str], ...]) -> ChatResponse:
        self.calls.append(messages)
        number = len(self.calls)
        content = f"Explanation\n```python\nimport numpy as np\ndef step_{number}():\n    return {number}\n```"
        return ChatResponse(content, None, "model", None, None, 0.1, {}, "stop")


def test_upstream_templates() -> None:
    assert sha256(_template(False).encode()).hexdigest() == (
        "9b407bdd44937f13cf090b5ae35436b7dc60a0fb8494caec6995c9c83cdadb03"
    )
    assert sha256(_template(True).encode()).hexdigest() == (
        "2c6946acf50f69d37647b83c04e77e6bdc5f3ff04e87d4aa6f79d0f27b5dcbc2"
    )


def test_upstream_fixed_code() -> None:
    hashes = {
        "13.6": "795a2b57c2d9bb12ca4eaf16d6b8e1f202015a89a886628858abf42a1b18a94e",
        "62.1": "bc9931d88a7d5950091b72a996a25b8be6c936fd136b01005e22c3d45b0008a2",
        "76.3": "4758300d96ea726cdc0fbf749f1bc437030d2a3e8b43bde232ecdeaf636cd367",
    }
    for step, digest in hashes.items():
        code = load_fixed(step)
        assert code is not None
        assert sha256(code.encode()).hexdigest() == digest


def test_prompt_context() -> None:
    prompt = render_prompt(
        problem(), 1, ("def step_1():\n    return 1",), with_background=True
    )

    assert "Solve step 1.\nBackground 1." in prompt
    assert "def step_1():\n    return 1" in prompt
    assert "Solve step 2.\nBackground 2." in prompt
    assert "------" not in prompt


@pytest.mark.asyncio
async def test_sequential_generation() -> None:
    client = FakeClient()
    results = await generate(problem(), client)

    assert len(client.calls) == 2
    assert "def step_1():\n    return 1" in client.calls[1][0]["content"]
    assert results[0].code == "\ndef step_1():\n    return 1\n"
    assert extract_code("text\n```\ndef value():\n    pass\n```") == (
        "\ndef value():\n    pass\n"
    )


@pytest.mark.asyncio
async def test_fixed_step_is_not_generated() -> None:
    original = problem()
    fixed = SciCodeStep(
        "62.1", "Fixed step.", "class Block:", "", "Fixed background.", ()
    )
    value = SciCodeProblem("62", original.dependencies, "", (fixed, original.steps[1]))
    client = FakeClient()

    results = await generate(value, client)

    assert len(client.calls) == 1
    assert results[0].source == "fixed"
    assert results[0].response is None
    assert load_fixed("62.1") in client.calls[0][0]["content"]
