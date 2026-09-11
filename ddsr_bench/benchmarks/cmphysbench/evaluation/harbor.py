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
from ddsr_bench.benchmarks.cmphysbench.data.schemas import ProblemSpec
from ddsr_bench.benchmarks.cmphysbench.evaluation.prepare import decode_instruction
from ddsr_bench.benchmarks.cmphysbench.generation.runner import generate
from ddsr_bench.generation.client import CLIENTS, ChatClient, ClientName, Sampling


class CMPhysBenchAgent(BaseAgent):
    """Generate one boxed CMPhysBench response from the Harbor host."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        sampling: Sampling | Mapping[str, Any] | None = None,
        stream: bool = False,
        api_key_env: str | None = None,
        client: ChatClient | None = None,
        client_name: ClientName = "vllm",
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
            "aliyun": "ALIYUN_API_KEY",
        }.get(client_name)
        self.client = client
        self.client_name = client_name

    @staticmethod
    @override
    def name() -> str:
        return "cmphysbench"

    @override
    def version(self) -> str:
        return __version__

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def _generate(self, problem: ProblemSpec) -> tuple[str, dict[str, Any]]:
        if self.client is not None:
            await self.client.preflight()
            return await generate(problem, self.client, self.sampling)
        async with CLIENTS[self.client_name](
            self.base_url,
            self.sampling,
            stream=self.stream,
            api_key=(self._get_env(self.api_key_env) if self.api_key_env else None),
        ) as client:
            await client.preflight()
            return await generate(problem, client, self.sampling)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        problem = decode_instruction(instruction)
        answer, record = await self._generate(problem)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "response.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as file:
            file.write(answer)
            file.flush()
            await environment.exec("mkdir -p /app", user="root")
            await environment.upload_file(file.name, "/app/answer.txt")

        usage = record["responses"][0]["usage"] or {}
        context.n_input_tokens = usage.get("prompt_tokens", 0)
        context.n_output_tokens = usage.get("completion_tokens", 0)
        context.metadata = {"problem_id": problem.id}
