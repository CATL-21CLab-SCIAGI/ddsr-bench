import asyncio
import json
from pathlib import Path

import httpx
import pytest

from ddsr_bench.generation.client import (
    AliyunClient,
    BedrockClient,
    ClientError,
    OpenAIClient,
    Sampling,
    VLLMClient,
)

MESSAGES = ({"role": "user", "content": "Solve."},)


@pytest.mark.asyncio
async def test_preflight_and_chat() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "solver"}]})
        return httpx.Response(
            200,
            json={
                "model": "solver",
                "choices": [
                    {
                        "message": {
                            "content": "answer",
                            "reasoning": "reasoning",
                        },
                        "finish_reason": "stop",
                        "stop_reason": "eos",
                    }
                ],
                "usage": {"completion_tokens": 3},
            },
        )

    sampling = Sampling(
        "solver",
        max_tokens=123,
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        seed=7,
        reasoning_effort="low",
    )
    async with VLLMClient(
        "http://localhost:8000/v1",
        sampling,
        timeout=9,
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.preflight()
        result = await client.chat(MESSAGES)

    payload = json.loads(requests[1].content)
    assert payload["messages"] == [{"role": "user", "content": "Solve."}]
    assert payload["max_tokens"] == 123
    assert payload["top_k"] == 20
    assert payload["seed"] == 7
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert requests[1].extensions["timeout"]["read"] == 9
    assert result.content == "answer"
    assert result.reasoning == "reasoning"
    assert result.usage == {"completion_tokens": 3}
    assert result.finish_reason == "stop"
    assert result.stop_reason == "eos"
    assert result.raw["model"] == "solver"


@pytest.mark.asyncio
async def test_retry_then_reject_malformed_response() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("not ready", request=request)
        return httpx.Response(200, json={"choices": []})

    async with VLLMClient(
        "http://localhost:8000/v1",
        Sampling("solver"),
        retries=1,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ClientError, match="requires a choice"):
            await client.chat(MESSAGES)

    assert calls == 2


@pytest.mark.asyncio
async def test_bedrock_preflight_without_model_listing() -> None:
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(404)

    async with BedrockClient(
        "https://example.com/openai/v1",
        Sampling("openai.gpt-5.6-luna"),
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.preflight()

    assert not called


@pytest.mark.asyncio
async def test_streaming_chat() -> None:
    request_body = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        request_body = json.loads(request.content)
        events = [
            {"model": "solver", "choices": [{"delta": {"reasoning": "why"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 2}},
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        body += "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    async with VLLMClient(
        "http://localhost:8000/v1",
        Sampling("solver"),
        stream=True,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.chat(MESSAGES)

    assert request_body["stream"] is True
    assert request_body["stream_options"] == {"include_usage": True}
    assert result.reasoning == "why"
    assert result.content == "answer"
    assert result.usage == {"completion_tokens": 2}
    assert result.finish_reason == "stop"
    assert len(result.raw["stream"]) == 4


@pytest.mark.asyncio
async def test_streaming_retry_before_data() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("not ready", request=request)
        body = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        body += 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        return httpx.Response(200, text=body + "data: [DONE]\n\n")

    async with VLLMClient(
        "http://localhost:8000/v1",
        Sampling("solver"),
        stream=True,
        retries=1,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.chat(MESSAGES)

    assert calls == 2
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_no_thinking() -> None:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        body = 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
        body += 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        return httpx.Response(200, text=body + "data: [DONE]\n\n")

    async with VLLMClient(
        "http://localhost:8000/v1",
        Sampling("solver", enable_thinking=False),
        stream=True,
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.chat(MESSAGES)

    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_openai_request() -> None:
    body = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal body
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]
            },
        )

    sampling = Sampling(
        "openai.gpt-5.6-luna",
        max_tokens=123,
        top_k=20,
        seed=7,
        reasoning_effort="low",
    )
    async with OpenAIClient(
        "https://example.com/openai/v1",
        sampling,
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.chat(MESSAGES)

    assert body == {
        "model": "openai.gpt-5.6-luna",
        "messages": [{"role": "user", "content": "Solve."}],
        "max_completion_tokens": 123,
        "reasoning_effort": "low",
    }

    async with OpenAIClient(
        "https://example.com/openai/v1",
        Sampling("openai.gpt-5.6-luna", reasoning_effort="none"),
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.chat(MESSAGES)

    assert body["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_aliyun_request() -> None:
    body = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal body
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]
            },
        )

    sampling = Sampling(
        "qwen3.8-max",
        max_tokens=123,
        temperature=0.6,
        top_p=0.95,
        seed=7,
        enable_thinking=False,
    )
    async with AliyunClient(
        "https://example.com/v1",
        sampling,
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.chat(MESSAGES)

    assert body == {
        "model": "qwen3.8-max",
        "messages": [{"role": "user", "content": "Solve."}],
        "max_tokens": 123,
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": 7,
        "enable_thinking": False,
    }


@pytest.mark.asyncio
async def test_requires_finish() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        body = 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
        return httpx.Response(200, text=body + "data: [DONE]\n\n")

    async with VLLMClient(
        "http://localhost:8000/v1",
        Sampling("solver"),
        stream=True,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ClientError, match="without a finish reason"):
            await client.chat(MESSAGES)


def test_thinking_options() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        Sampling("solver", reasoning_effort="low", enable_thinking=False)


@pytest.mark.asyncio
async def test_dynamic_budget_and_per_request_cap_are_concurrency_safe() -> None:
    bodies = []

    async def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            assert body["chat_template_kwargs"] == {"reasoning_effort": "xhigh"}
            assert body["add_generation_prompt"] is True
            await asyncio.sleep(0)
            return httpx.Response(200, json={"count": 4000, "max_model_len": 262144})
        bodies.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "answer"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async with VLLMClient(
        "http://localhost:8000/v1",
        Sampling("solver", max_tokens=262144, reasoning_effort="xhigh"),
        context_window=262144,
        context_safety_tokens=32,
        transport=httpx.MockTransport(handler),
    ) as client:
        first, second = await asyncio.gather(
            client.chat(MESSAGES, seed=101),
            client.chat(MESSAGES, max_tokens=65536, seed=202),
        )
        assert client.sampling.max_tokens == 262144
        assert client.sampling.seed is None
    assert sorted(b["max_tokens"] for b in bodies) == [65536, 258112]
    assert first.request_budget["prompt_tokens"] == 4000
    assert second.request_budget["max_tokens"] == 65536
    assert {(b["seed"], b["max_tokens"]) for b in bodies} == {
        (101, 258112),
        (202, 65536),
    }
    assert (first.seed, second.seed) == (101, 202)


@pytest.mark.asyncio
@pytest.mark.parametrize("count, expected", [(250000, 12112), (262112, None)])
async def test_formatting_budget_respects_remaining_context(count, expected) -> None:
    bodies = []

    def handler(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": count, "max_model_len": 262144})
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "answer"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async with VLLMClient(
        "http://localhost/v1",
        Sampling("solver"),
        context_window=262144,
        transport=httpx.MockTransport(handler),
    ) as client:
        if expected is None:
            with pytest.raises(ClientError, match="exhausts context"):
                await client.chat(MESSAGES, max_tokens=65536)
            assert not bodies
        else:
            await client.chat(MESSAGES, max_tokens=65536)
            assert bodies[0]["max_tokens"] == expected


@pytest.mark.asyncio
async def test_stream_journal_retains_partial_output_on_failure(tmp_path: Path) -> None:
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning":"partial thought"}}]}\n\n'
            raise httpx.ReadError("disconnected")

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=BrokenStream())

    journal = tmp_path / "stream.jsonl"
    async with VLLMClient(
        "http://localhost/v1",
        Sampling("solver"),
        stream=True,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ClientError, match="stream failed"):
            await client.chat(MESSAGES, stream_path=journal)
    assert calls == 1  # Never silently retry an already-started answer.
    lines = [json.loads(line) for line in journal.read_text().splitlines()]
    assert lines[0]["budget"]["max_tokens"] == 32768
    assert lines[1]["choices"][0]["delta"]["reasoning"] == "partial thought"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile,token_key", [("native", "max_tokens"), ("openai", "max_completion_tokens")]
)
async def test_aliyun_reasoning_profiles_preserve_requests_and_trial_seeds(
    profile, token_key
):
    bodies = []

    async def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        await asyncio.sleep(0)
        event = {"choices": [{"delta": {"content": "42"}, "finish_reason": "stop"}]}
        return httpx.Response(
            200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
        )

    sampling = Sampling(
        "deepseek-v4-flash-0731",
        max_tokens=393216,
        temperature=1,
        top_p=1,
        reasoning_effort="max",
    )
    async with AliyunClient(
        "https://example.com/v1",
        sampling,
        request_profile=profile,
        stream=True,
        transport=httpx.MockTransport(handler),
    ) as client:
        results = await asyncio.gather(
            client.chat(MESSAGES, seed=42),
            client.chat(MESSAGES, seed=43, max_tokens=131072),
        )
    assert {b[token_key] for b in bodies} == {393216, 131072}
    assert {b["seed"] for b in bodies} == {42, 43}
    assert all(b["reasoning_effort"] == "max" for b in bodies)
    assert [r.seed for r in results] == [42, 43]
    if profile == "openai":
        expected = OpenAIClient("https://example.com/v1", sampling)
        try:
            legacy = expected._payload(MESSAGES)
            assert bodies[0] == legacy | {
                "seed": 42,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            assert "temperature" not in bodies[0] and "top_p" not in bodies[0]
        finally:
            await expected.http.aclose()


def test_aliyun_rejects_incompatible_profiles():
    with pytest.raises(ValueError, match="request_profile"):
        AliyunClient(
            "https://example.com/v1", Sampling("solver"), request_profile="guess"
        )
