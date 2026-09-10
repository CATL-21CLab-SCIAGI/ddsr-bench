"""Check an OpenAI-compatible model endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from ddsr_bench.generation.client import (
    CLIENTS,
    ChatClient,
    ChatResponse,
    ClientName,
    Sampling,
)

from .utils import read_api_key


async def smoke(client: ChatClient) -> ChatResponse:
    """Check model discovery and one OpenAI-compatible chat completion."""
    await client.preflight()
    return await client.chat(
        ({"role": "user", "content": "Reply with exactly: ready /no_think"},)
    )


async def _run(
    base_url: str,
    model: str,
    api_key_env: str | None,
    client_name: ClientName,
) -> ChatResponse:
    sampling = Sampling(
        model=model,
        max_tokens=128,
        temperature=0.0,
        reasoning_effort="none" if client_name in ("openai", "bedrock") else None,
        enable_thinking=False if client_name == "vllm" else None,
    )
    async with CLIENTS[client_name](
        base_url,
        sampling,
        timeout=120,
        api_key=read_api_key(api_key_env),
    ) as client:
        return await smoke(client)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a CritPt model endpoint")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key-env")
    parser.add_argument("--client", choices=tuple(CLIENTS), default="vllm")
    parser.add_argument("--model", default="ddsr-local")
    args = parser.parse_args()
    response = asyncio.run(
        _run(args.base_url, args.model, args.api_key_env, args.client)
    )
    print(
        json.dumps(
            {
                "model": response.model,
                "content": response.content,
                "latency": response.latency,
                "usage": response.usage,
            },
            indent=2,
        )
    )
