from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from ddsr_bench import __version__
from ddsr_bench.benchmarks.critpt.evaluation.prepare import decode_instruction
from ddsr_bench.benchmarks.critpt.generation.prompts import PromptStyle
from ddsr_bench.benchmarks.critpt.generation.runner import generate
from ddsr_bench.generation.client import (
    CLIENTS,
    ChatClient,
    ClientName,
    Sampling,
)
from ddsr_bench.grading.validation import extract_answer


class CritPtAgent(BaseAgent):
    def __init__(
        self,
        logs_dir: Path,
        model_name: str,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        style: PromptStyle = "one-step",
        sampling: Sampling | Mapping[str, Any] | None = None,
        stream: bool = False,
        api_key_env: str | None = None,
        client: ChatClient | None = None,
        client_name: ClientName = "vllm",
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, model_name, **kwargs)
        self.base_url = base_url
        self.style = style
        self.sampling = (
            sampling
            if isinstance(sampling, Sampling)
            else Sampling(model_name, **dict(sampling or {}))
        )
        self.stream = stream
        self.api_key_env = api_key_env or {
            "openai": "OPENAI_API_KEY",
            "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
        }.get(client_name)
        self.client = client
        self.client_name = client_name

    @staticmethod
    @override
    def name() -> str:
        return "critpt"

    @override
    def version(self) -> str:
        return __version__

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        problem = decode_instruction(instruction)
        if self.client is None:
            async with CLIENTS[self.client_name](
                self.base_url,
                self.sampling,
                stream=self.stream,
                api_key=(self._get_env(self.api_key_env) if self.api_key_env else None),
            ) as client:
                await client.preflight()
                answer, record = await generate(
                    client, problem, self.style, self.sampling
                )
        else:
            await self.client.preflight()
            answer, record = await generate(
                self.client, problem, self.style, self.sampling
            )

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "response.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        code = extract_answer(answer, problem.code_template)
        with NamedTemporaryFile("w", suffix=".py", encoding="utf-8") as file:
            file.write(code)
            file.flush()
            await environment.exec("mkdir -p /app", user="root")
            await environment.upload_file(file.name, "/app/answer.py")

        usage = [response["usage"] or {} for response in record["responses"]]
        context.n_input_tokens = sum(item.get("prompt_tokens", 0) for item in usage)
        context.n_output_tokens = sum(
            item.get("completion_tokens", 0) for item in usage
        )
        context.metadata = {"problem_id": problem.id, "strategy": self.style}
