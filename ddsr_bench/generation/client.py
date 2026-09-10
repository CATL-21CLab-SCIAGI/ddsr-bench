from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol, Self

import httpx

ClientName = Literal["vllm", "openai", "bedrock"]
ChatMessages = tuple[dict[str, str], ...]


class ClientError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Sampling:
    model: str
    max_tokens: int = 32_768
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None
    reasoning_effort: (
        Literal["none", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    enable_thinking: bool | None = None

    def __post_init__(self) -> None:
        if self.reasoning_effort is not None and self.enable_thinking is not None:
            raise ValueError(
                "reasoning effort and enable thinking are mutually exclusive"
            )


@dataclass(frozen=True, slots=True)
class ChatResponse:
    content: str
    reasoning: str | None
    model: str
    usage: dict[str, Any] | None
    seed: int | None
    latency: float
    raw: dict[str, Any]
    finish_reason: str | None = None
    stop_reason: str | int | None = None


class ChatClient(Protocol):
    """Typing protocol for clients; it is not instantiated at runtime."""

    async def preflight(self) -> None: ...

    async def chat(self, messages: ChatMessages) -> ChatResponse: ...


class BaseClient(ABC):
    """Shared transport and response handling for chat clients."""

    def __init__(
        self,
        base_url: str,
        sampling: Sampling,
        *,
        timeout: float = 1_200,
        retries: int = 2,
        api_key: str | None = None,
        stream: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self.sampling = sampling
        self.retries = retries
        self.stream = stream
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.http.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(self.retries + 1):
            try:
                response = await self.http.request(method, path, **kwargs)
                response.raise_for_status()
                return response
            except httpx.TransportError as error:
                if attempt == self.retries:
                    raise ClientError(
                        f"{method} {path} failed after retries"
                    ) from error
        raise AssertionError("unreachable")

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as error:
            raise ClientError("server returned invalid JSON") from error
        if not isinstance(data, dict):
            raise ClientError("server response must be a JSON object")
        return data

    async def preflight(self) -> None:
        data = self._json(await self._request("GET", "models"))
        models = data.get("data")
        ids = {
            model.get("id")
            for model in models or []
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        }
        if self.sampling.model not in ids:
            raise ClientError(f"model {self.sampling.model!r} is not served")

    @abstractmethod
    def _payload(self, messages: ChatMessages) -> dict[str, Any]: ...

    async def chat(self, messages: ChatMessages) -> ChatResponse:
        payload = self._payload(messages)
        if self.stream:
            return await self._stream(payload)
        started = perf_counter()
        data = self._json(await self._request("POST", "chat/completions", json=payload))
        latency = perf_counter() - started
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ClientError("server response requires a choice")
        choice = choices[0]
        finish_reason = (
            choice.get("finish_reason") if isinstance(choice, dict) else None
        )
        if not isinstance(finish_reason, str):
            raise ClientError("server response requires a finish reason")
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        reasoning = message.get("reasoning") if isinstance(message, dict) else None
        if reasoning is None and isinstance(message, dict):
            # Backward compatibility with older OpenAI-compatible servers.
            reasoning = message.get("reasoning_content")
        if not isinstance(content, str):
            raise ClientError("server response requires text content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ClientError("server reasoning content must be text")
        return ChatResponse(
            content=content,
            reasoning=reasoning,
            model=(
                data["model"]
                if isinstance(data.get("model"), str)
                else self.sampling.model
            ),
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
            seed=self.sampling.seed,
            latency=latency,
            raw=data,
            finish_reason=finish_reason,
            stop_reason=choice.get("stop_reason"),
        )

    async def _stream(self, payload: dict[str, Any]) -> ChatResponse:
        payload |= {"stream": True, "stream_options": {"include_usage": True}}
        chunks: list[dict[str, Any]] = []
        content: list[str] = []
        reasoning: list[str] = []
        usage = None
        model = self.sampling.model
        finish_reason = None
        stop_reason = None
        started = perf_counter()

        for attempt in range(self.retries + 1):
            try:
                async with self.http.stream(
                    "POST", "chat/completions", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as error:
                            raise ClientError(
                                "server returned invalid stream JSON"
                            ) from error
                        if not isinstance(chunk, dict):
                            raise ClientError(
                                "server stream chunk must be a JSON object"
                            )
                        chunks.append(chunk)
                        if isinstance(chunk.get("model"), str):
                            model = chunk["model"]
                        if isinstance(chunk.get("usage"), dict):
                            usage = chunk["usage"]
                        choices = chunk.get("choices")
                        choice = (
                            choices[0] if isinstance(choices, list) and choices else {}
                        )
                        terminal = (
                            choice.get("finish_reason")
                            if isinstance(choice, dict)
                            else None
                        )
                        if terminal is not None:
                            if not isinstance(terminal, str):
                                raise ClientError(
                                    "server returned invalid finish reason"
                                )
                            finish_reason = terminal
                            stop_reason = choice.get("stop_reason")
                        delta = choice.get("delta") if isinstance(choice, dict) else {}
                        if not isinstance(delta, dict):
                            continue
                        if isinstance(delta.get("content"), str):
                            content.append(delta["content"])
                        reason = delta.get("reasoning")
                        if reason is None:
                            # Backward compatibility with older servers.
                            reason = delta.get("reasoning_content")
                        if isinstance(reason, str):
                            reasoning.append(reason)
                break
            except httpx.TransportError as error:
                if chunks or attempt == self.retries:
                    raise ClientError("POST chat/completions stream failed") from error
            except httpx.HTTPError as error:
                raise ClientError("POST chat/completions stream failed") from error

        if finish_reason is None:
            raise ClientError("server stream ended without a finish reason")
        return ChatResponse(
            content="".join(content),
            reasoning="".join(reasoning) or None,
            model=model,
            usage=usage,
            seed=self.sampling.seed,
            latency=perf_counter() - started,
            raw={"stream": chunks},
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )


class VLLMClient(BaseClient):
    """Chat client using vLLM request extensions."""

    def _payload(self, messages: ChatMessages) -> dict[str, Any]:
        payload = {
            "model": self.sampling.model,
            "messages": list(messages),
            "max_tokens": self.sampling.max_tokens,
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
        }
        if self.sampling.top_k is not None:
            payload["top_k"] = self.sampling.top_k
        if self.sampling.seed is not None:
            payload["seed"] = self.sampling.seed
        template_kwargs = {
            key: value
            for key, value in (
                ("reasoning_effort", self.sampling.reasoning_effort),
                ("enable_thinking", self.sampling.enable_thinking),
            )
            if value is not None
        }
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs
        return payload


class OpenAIClient(BaseClient):
    """Chat client using the OpenAI request schema."""

    def _payload(self, messages: ChatMessages) -> dict[str, Any]:
        payload = {
            "model": self.sampling.model,
            "messages": list(messages),
            "max_completion_tokens": self.sampling.max_tokens,
        }
        if self.sampling.reasoning_effort is not None:
            payload["reasoning_effort"] = self.sampling.reasoning_effort
        return payload


class BedrockClient(OpenAIClient):
    """OpenAI-compatible Bedrock client without model discovery."""

    async def preflight(self) -> None:
        pass


CLIENTS: dict[ClientName, type[BaseClient]] = {
    "vllm": VLLMClient,
    "openai": OpenAIClient,
    "bedrock": BedrockClient,
}
