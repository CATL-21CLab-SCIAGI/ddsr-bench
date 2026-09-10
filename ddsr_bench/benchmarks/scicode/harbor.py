from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from ddsr_bench import __version__
from ddsr_bench.benchmarks.scicode.schemas import SciCodeProblem
from ddsr_bench.generation.client import (
    CLIENTS,
    ChatClient,
    ClientName,
    Sampling,
)

from .generation import StepGeneration, generate
from .prepare import decode_instruction


class SciCodeAgent(BaseAgent):
    """Generate one SciCode problem sequentially from the Harbor host."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        sampling: Sampling | dict[str, Any] | None = None,
        stream: bool = False,
        api_key_env: str | None = None,
        client: ChatClient | None = None,
        client_name: ClientName = "vllm",
        with_background: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, model_name, **kwargs)
        self.base_url = base_url
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
        self.with_background = with_background

    @staticmethod
    @override
    def name() -> str:
        return "scicode"

    @override
    def version(self) -> str:
        return __version__

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def _generate(self, problem: SciCodeProblem) -> tuple[StepGeneration, ...]:
        if self.client is not None:
            await self.client.preflight()
            return await generate(
                problem, self.client, with_background=self.with_background
            )
        async with CLIENTS[self.client_name](
            self.base_url,
            self.sampling,
            stream=self.stream,
            api_key=(self._get_env(self.api_key_env) if self.api_key_env else None),
        ) as client:
            await client.preflight()
            return await generate(problem, client, with_background=self.with_background)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        problem = decode_instruction(instruction)
        steps = await self._generate(problem)
        solution = "\n\n".join((problem.dependencies, *(step.code for step in steps)))

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "problem_id": problem.id,
            "with_background": self.with_background,
            "steps": [asdict(step) for step in steps],
        }
        (self.logs_dir / "response.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with NamedTemporaryFile("w", suffix=".py", encoding="utf-8") as file:
            file.write(solution)
            file.flush()
            await environment.exec("mkdir -p /app", user="root")
            await environment.upload_file(file.name, "/app/solution.py")

        usage = [step.response.usage or {} for step in steps if step.response]
        context.n_input_tokens = sum(item.get("prompt_tokens", 0) for item in usage)
        context.n_output_tokens = sum(
            item.get("completion_tokens", 0) for item in usage
        )
        context.metadata = {"problem_id": problem.id, "steps": len(steps)}
