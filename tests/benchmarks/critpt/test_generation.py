from pathlib import Path

import pytest

from ddsr_bench.benchmarks.critpt.data.schemas import Challenge, Problem, ProblemSpec
from ddsr_bench.benchmarks.critpt.generation.prompts import parse_prompt, system_prompt
from ddsr_bench.benchmarks.critpt.generation.runner import (
    Message,
    converse,
    converse_challenge,
)

PROBLEM = ProblemSpec(
    id="p1",
    type="main",
    index=None,
    statement="Find the result.",
    code_template="def answer():\n    return ...",
    source="critpt",
    source_path=Path("problem.json"),
)


class FakeModel:
    def __init__(self, *responses: str) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[Message, ...]] = []

    async def complete(self, messages: tuple[Message, ...]) -> str:
        self.calls.append(messages)
        return next(self.responses)


@pytest.mark.asyncio
async def test_one_step() -> None:
    model = FakeModel("final code")

    result = await converse(PROBLEM, "one-step", model.complete)

    assert len(model.calls) == 1
    assert model.calls[0] == (
        Message("system", system_prompt("one-step")),
        Message(
            "user",
            f"{PROBLEM.statement}\n\n```python\n{PROBLEM.code_template}\n```",
        ),
    )
    assert result.content == "final code"
    assert result.messages[-1] == Message("assistant", "final code")


@pytest.mark.asyncio
async def test_two_step() -> None:
    model = FakeModel("reasoned answer", "formatted code")

    result = await converse(PROBLEM, "two-step", model.complete)

    assert len(model.calls) == 2
    assert model.calls[0] == (
        Message("system", system_prompt("two-step")),
        Message("user", PROBLEM.statement),
    )
    assert model.calls[1][-2:] == (
        Message("assistant", "reasoned answer"),
        Message("user", parse_prompt(PROBLEM.code_template)),
    )
    assert PROBLEM.code_template not in model.calls[0][1].content
    assert result.content == "formatted code"


def challenge() -> Challenge:
    def problem(problem_id: str, problem_type: str, index: int | None) -> Problem:
        return Problem(
            ProblemSpec(
                problem_id,
                problem_type,
                index,
                f"Solve {problem_id}.",
                "def answer():\n    return ...",
                "critpt",
                Path("challenge.json"),
            ),
            None,
        )

    return Challenge(
        "challenge",
        (
            problem("main", "main", None),
            problem("sub_0", "sub", 0),
            problem("sub_1", "sub", 1),
        ),
        Path("challenge.json"),
    )


@pytest.mark.asyncio
async def test_challenge_carries_generated_subproblem_context() -> None:
    model = FakeModel(
        "main reasoning",
        "main code",
        "sub 0 reasoning",
        "sub 0 code",
        "sub 1 reasoning",
        "sub 1 code",
    )

    results = await converse_challenge(challenge(), "two-step", model.complete)

    assert [result.content for result in results] == [
        "main code",
        "sub 0 code",
        "sub 1 code",
    ]
    sub_1_request = model.calls[4]
    assert Message("assistant", "sub 0 reasoning") in sub_1_request
    assert Message("assistant", "sub 0 code") not in sub_1_request
    assert Message("assistant", "main reasoning") not in sub_1_request


@pytest.mark.asyncio
async def test_challenge_uses_stored_golden_context_independently() -> None:
    model = FakeModel("main", "sub 0", "sub 1")

    await converse_challenge(challenge(), "one-step", model.complete, use_golden=True)

    assert all(len(request) == 2 for request in model.calls)
