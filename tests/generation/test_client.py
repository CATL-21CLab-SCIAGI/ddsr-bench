import json

import httpx
import pytest

from ddsr_bench.generation.client import (
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
