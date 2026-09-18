from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol, Self

import httpx

ClientName = Literal["vllm", "openai", "bedrock", "aliyun"]
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
    request_budget: dict[str, int] | None = None


class ChatClient(Protocol):
    """Typing protocol for clients; it is not instantiated at runtime."""

    async def preflight(self) -> None: ...

    async def chat(
        self,
        messages: ChatMessages,
        *,
        max_tokens: int | None = None,
        stream_path: Path | None = None,
        seed: int | None = None,
    ) -> ChatResponse: ...


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

    async def _prepare_payload(
        self,
        messages: ChatMessages,
        max_tokens: int | None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        payload = self._payload(messages)
        key = "max_tokens" if "max_tokens" in payload else "max_completion_tokens"
        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            payload[key] = max_tokens
        return payload, {"max_tokens": payload[key]}

    async def chat(
        self,
        messages: ChatMessages,
        *,
        max_tokens: int | None = None,
        stream_path: Path | None = None,
        seed: int | None = None,
    ) -> ChatResponse:
        payload, budget = await self._prepare_payload(messages, max_tokens)
        if seed is not None:
            payload["seed"] = seed
        request_seed = payload.get("seed", self.sampling.seed)
        if self.stream:
            response = await self._stream(payload, stream_path, budget)
            return replace(response, request_budget=budget, seed=request_seed)
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
            seed=request_seed,
            latency=latency,
            raw=data,
            finish_reason=finish_reason,
            stop_reason=choice.get("stop_reason"),
            request_budget=budget,
        )

    async def _stream(
        self,
        payload: dict[str, Any],
        stream_path: Path | None = None,
        budget: dict[str, int] | None = None,
    ) -> ChatResponse:
        # A journal retains partial output on disconnect and avoids keeping every
        # SSE dictionary in RAM during many concurrent long-context generations.
        context = (
            stream_path.open("x", encoding="utf-8") if stream_path else nullcontext()
        )
        with context as journal:
            if journal is not None:
                journal.write(
                    json.dumps(
                        {"request": payload, "budget": budget}, ensure_ascii=False
                    )
                    + "\n"
                )
                journal.flush()
            return await self._stream_response(payload, journal, stream_path)

    async def _stream_response(self, payload, journal, stream_path) -> ChatResponse:
        payload |= {"stream": True, "stream_options": {"include_usage": True}}
        chunks: list[dict[str, Any]] = []
        content: list[str] = []
        reasoning: list[str] = []
        usage = None
        model = self.sampling.model
        finish_reason = None
        stop_reason = None
        started = perf_counter()
        flushed_at = started
        received = False

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
                        received = True
                        if journal is None:
                            chunks.append(chunk)
                        else:
                            journal.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                            if perf_counter() - flushed_at >= 1:
                                journal.flush()
                                flushed_at = perf_counter()
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
                if received or attempt == self.retries:
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
            raw=(
                {"stream_file": str(stream_path)} if stream_path else {"stream": chunks}
            ),
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )


class VLLMClient(BaseClient):
    """Chat client using vLLM request extensions."""

    def __init__(
        self,
        base_url: str,
        sampling: Sampling,
        *,
        context_window: int | None = None,
        context_safety_tokens: int = 32,
        **kwargs: Any,
    ) -> None:
        if context_window is not None and context_window <= 0:
            raise ValueError("context_window must be positive")
        if context_safety_tokens < 0:
            raise ValueError("context_safety_tokens cannot be negative")
        super().__init__(base_url, sampling, **kwargs)
        self.context_window = context_window
        self.context_safety_tokens = context_safety_tokens

    async def _prepare_payload(
        self,
        messages: ChatMessages,
        max_tokens: int | None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        payload, budget = await super()._prepare_payload(messages, max_tokens)
        if self.context_window is None:
            return payload, budget
        request = {
            "model": payload["model"],
            "messages": payload["messages"],
            "add_generation_prompt": True,
            "chat_template_kwargs": payload.get("chat_template_kwargs", {}),
        }
        url = str(self.http.base_url.copy_with(path="/tokenize"))
        tokenized = self._json(await self._request("POST", url, json=request))
        count = tokenized.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ClientError("tokenizer response requires a nonnegative token count")
        limit = min(
            self.context_window, tokenized.get("max_model_len", self.context_window)
        )
        available = limit - count - self.context_safety_tokens
        if available <= 0:
            raise ClientError(
                f"input of {count} tokens exhausts context window {limit}"
            )
        payload["max_tokens"] = min(payload["max_tokens"], available)
        return payload, {
            "context_window": limit,
            "prompt_tokens": count,
            "safety_tokens": self.context_safety_tokens,
            "requested_max_tokens": budget["max_tokens"],
            "max_tokens": payload["max_tokens"],
        }

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


class AliyunClient(BaseClient):
    """Chat client using Alibaba Cloud's OpenAI-compatible parameters."""

    def __init__(
        self,
        base_url: str,
        sampling: Sampling,
        *,
        request_profile: Literal["native", "openai"] = "native",
        **kwargs: Any,
    ) -> None:
        if request_profile not in ("native", "openai"):
            raise ValueError("request_profile must be native or openai")
        super().__init__(base_url, sampling, **kwargs)
        self.request_profile = request_profile

    def _payload(self, messages: ChatMessages) -> dict[str, Any]:
        if self.sampling.top_k is not None:
            raise ValueError("Aliyun does not document top_k for this endpoint")
        if self.request_profile == "openai":
            if self.sampling.enable_thinking is not None:
                raise ValueError("enable_thinking requires the native request profile")
            return OpenAIClient._payload(self, messages)
        payload = {
            "model": self.sampling.model,
            "messages": list(messages),
            "max_tokens": self.sampling.max_tokens,
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
        }
        if self.sampling.seed is not None:
            payload["seed"] = self.sampling.seed
        if self.sampling.enable_thinking is not None:
            payload["enable_thinking"] = self.sampling.enable_thinking
        if self.sampling.reasoning_effort is not None:
            payload["reasoning_effort"] = self.sampling.reasoning_effort
        return payload


CLIENTS: dict[ClientName, type[BaseClient]] = {
    "vllm": VLLMClient,
    "openai": OpenAIClient,
    "bedrock": BedrockClient,
    "aliyun": AliyunClient,
}
