import json
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


@pytest.mark.asyncio
async def test_stage_one_survives_stage_two_failure(tmp_path: Path) -> None:
    from ddsr_bench.benchmarks.critpt.generation.runner import generate
    from ddsr_bench.generation.client import ChatResponse, Sampling

    class Client:
        calls = 0

        async def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert "max_tokens" not in kwargs
                return ChatResponse(
                    "first answer", "reasoning", "solver", None, None, 1, {}, "stop"
                )
            checkpoint = json.loads((tmp_path / "stage-1.json").read_text())
            assert checkpoint["responses"][0]["reasoning"] == "reasoning"
            assert kwargs["max_tokens"] == 65536
            assert messages[-2] == {"role": "assistant", "content": "first answer"}
            raise RuntimeError("formatting disconnected")

    with pytest.raises(RuntimeError, match="formatting disconnected"):
        await generate(
            Client(),
            PROBLEM,
            "two-step",
            Sampling("solver"),
            formatting_max_tokens=65536,
            checkpoint_dir=tmp_path,
        )
    assert (tmp_path / "response.json").is_file()
    assert not (tmp_path / "stage-2.json").exists()


@pytest.mark.asyncio
async def test_truncation_is_saved_and_not_passed_to_formatting(tmp_path: Path) -> None:
    from ddsr_bench.benchmarks.critpt.generation.runner import generate
    from ddsr_bench.generation.client import ChatResponse, Sampling

    class Client:
        calls = 0

        async def chat(self, messages, **kwargs):
            self.calls += 1
            return ChatResponse(
                "unfinished", "thought", "solver", None, None, 1, {}, "length"
            )

    client = Client()
    with pytest.raises(ValueError, match="did not finish normally: length"):
        await generate(
            client,
            PROBLEM,
            "two-step",
            Sampling("solver"),
            checkpoint_dir=tmp_path,
            require_complete_stages=True,
        )
    assert client.calls == 1
    assert (
        json.loads((tmp_path / "stage-1.json").read_text())["responses"][0][
            "finish_reason"
        ]
        == "length"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("formatting_cap", [65536, 131072])
async def test_resume_reuses_stage_one_and_preserves_interrupted_stream(
    tmp_path, formatting_cap
):
    from ddsr_bench.benchmarks.critpt.generation.runner import generate
    from ddsr_bench.generation.client import ChatResponse, Sampling

    class Interrupted:
        calls = 0

        async def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return ChatResponse(
                    "derivation", "thought", "solver", None, 42, 1, {}, "stop"
                )
            kwargs["stream_path"].write_text("partial formatting stream\n")
            raise RuntimeError("interrupted")

    sampling = Sampling("solver", seed=42)
    with pytest.raises(RuntimeError, match="interrupted"):
        await generate(
            Interrupted(),
            PROBLEM,
            "two-step",
            sampling,
            formatting_max_tokens=65536,
            checkpoint_dir=tmp_path,
        )

    class Resumed:
        calls = 0

        async def chat(self, messages, **kwargs):
            self.calls += 1
            assert messages[-2] == {"role": "assistant", "content": "derivation"}
            assert kwargs["seed"] == 42 and kwargs["max_tokens"] == formatting_cap
            assert not kwargs["stream_path"].exists()
            return ChatResponse("final code", None, "solver", None, 42, 1, {}, "stop")

    client = Resumed()
    answer, record = await generate(
        client,
        PROBLEM,
        "two-step",
        sampling,
        formatting_max_tokens=formatting_cap,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    assert client.calls == 1 and answer == "final code"
    assert len(record["responses"]) == 2
    assert record["formatting_max_tokens"] == formatting_cap
    archived = list(tmp_path.glob("stage-2.interrupted-*.stream.jsonl"))
    assert (
        len(archived) == 1 and archived[0].read_text() == "partial formatting stream\n"
    )
    with pytest.raises(ValueError, match="formatting_max_tokens changed"):
        await generate(
            client,
            PROBLEM,
            "two-step",
            sampling,
            formatting_max_tokens=formatting_cap + 1,
            checkpoint_dir=tmp_path,
            resume=True,
        )
    assert client.calls == 1
    with pytest.raises(ValueError, match="sampling changed"):
        await generate(
            client,
            PROBLEM,
            "two-step",
            Sampling("solver", seed=43),
            formatting_max_tokens=65536,
            checkpoint_dir=tmp_path,
            resume=True,
        )
