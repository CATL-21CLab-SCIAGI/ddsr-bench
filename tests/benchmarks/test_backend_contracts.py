import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from harbor.models.agent.context import AgentContext

from ddsr_bench.benchmarks.cmphysbench.data.schemas import (
    AnswerSpec as CMAnswer,
)
from ddsr_bench.benchmarks.cmphysbench.data.schemas import Problem as CMProblem
from ddsr_bench.benchmarks.cmphysbench.data.schemas import ProblemSpec as CMSpec
from ddsr_bench.benchmarks.cmphysbench.evaluation import static as cm_static
from ddsr_bench.benchmarks.cmphysbench.evaluation.harbor import CMPhysBenchAgent
from ddsr_bench.benchmarks.cmphysbench.evaluation.prepare import (
    encode_instruction as encode_cm,
)
from ddsr_bench.benchmarks.cmtbench.data.schemas import AnswerSpec as CMTAnswer
from ddsr_bench.benchmarks.cmtbench.data.schemas import Problem as CMTProblem
from ddsr_bench.benchmarks.cmtbench.data.schemas import ProblemSpec as CMTSpec
from ddsr_bench.benchmarks.cmtbench.evaluation import static as cmt_static
from ddsr_bench.benchmarks.cmtbench.evaluation.harbor import CMTBenchAgent
from ddsr_bench.benchmarks.cmtbench.evaluation.prepare import (
    encode_instruction as encode_cmt,
)
from ddsr_bench.benchmarks.phybench.data.schemas import AnswerSpec as PHYAnswer
from ddsr_bench.benchmarks.phybench.data.schemas import Problem as PHYProblem
from ddsr_bench.benchmarks.phybench.data.schemas import ProblemSpec as PHYSpec
from ddsr_bench.benchmarks.phybench.evaluation import static as phy_static
from ddsr_bench.benchmarks.phybench.evaluation.harbor import PHYBenchAgent
from ddsr_bench.benchmarks.phybench.evaluation.prepare import (
    encode_instruction as encode_phy,
)
from ddsr_bench.benchmarks.prismphysics.data.schemas import (
    AnswerSpec as PRISMAnswer,
)
from ddsr_bench.benchmarks.prismphysics.data.schemas import FormulaNode
from ddsr_bench.benchmarks.prismphysics.data.schemas import Problem as PRISMProblem
from ddsr_bench.benchmarks.prismphysics.data.schemas import ProblemSpec as PRISMSpec
from ddsr_bench.benchmarks.prismphysics.evaluation import static as prism_static
from ddsr_bench.benchmarks.prismphysics.evaluation.harbor import PRISMPhysicsAgent
from ddsr_bench.benchmarks.prismphysics.evaluation.prepare import (
    encode_instruction as encode_prism,
)
from ddsr_bench.generation.client import ChatResponse, Sampling


class FakeClient:
    async def preflight(self) -> None:
        pass

    async def chat(self, messages) -> ChatResponse:
        return ChatResponse(
            "public answer",
            "public reasoning",
            "teacher",
            {"prompt_tokens": 3, "completion_tokens": 2},
            7,
            0.1,
            {"id": "response"},
            "stop",
        )


class FakeEnvironment:
    answer = ""

    async def exec(self, command: str, *, user: str) -> None:
        assert (command, user) == ("mkdir -p /app", "root")

    async def upload_file(self, source_path: str, target_path: str) -> None:
        assert target_path == "/app/answer.txt"
        self.answer = Path(source_path).read_text(encoding="utf-8")


@dataclass(frozen=True)
class Case:
    name: str
    problem: Any
    agent: type
    encode: Any
    static: Any


CASES = (
    Case(
        "cmphysbench",
        CMProblem(
            CMSpec("cm", "", "question", "x", "Expression", "topic"),
            CMAnswer("SECRET_CM"),
        ),
        CMPhysBenchAgent,
        encode_cm,
        cm_static,
    ),
    Case(
        "phybench",
        PHYProblem(
            PHYSpec("phy", "MECHANICS", "question"), PHYAnswer("SECRET_PHY", "solution")
        ),
        PHYBenchAgent,
        encode_phy,
        phy_static,
    ),
    Case(
        "cmtbench",
        CMTProblem(
            CMTSpec("cmt", "question", "Other"), CMTAnswer("SECRET_CMT", "", "")
        ),
        CMTBenchAgent,
        encode_cmt,
        cmt_static,
    ),
    Case(
        "prismphysics",
        PRISMProblem(
            PRISMSpec("01-1", 1, "01", "mechanics", "question", "source", (), ()),
            PRISMAnswer((FormulaNode(1, "SECRET_PRISM=x", (), True),)),
        ),
        PRISMPhysicsAgent,
        encode_prism,
        prism_static,
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_static_and_harbor_generation_match(case, tmp_path, monkeypatch) -> None:
    """Both backends must publish the same public generation and no reference."""
    sampling = Sampling("teacher", seed=7)
    harbor_dir = tmp_path / "harbor"
    environment = FakeEnvironment()
    agent = case.agent(harbor_dir, "teacher", client=FakeClient(), sampling=sampling)
    await agent.run(case.encode(case.problem.spec), environment, AgentContext())

    monkeypatch.setattr(
        case.static,
        "verify",
        lambda *args: {"reward": 1.0, "mode": "test", "status": "passed"},
    )
    static_dir = tmp_path / "static"
    await case.static.run_trial(
        case.problem,
        0,
        static_dir,
        FakeClient(),
        sampling,
        client_name="vllm",
    )

    harbor_record = json.loads((harbor_dir / "response.json").read_text())
    static_record = json.loads((static_dir / "agent" / "response.json").read_text())
    assert static_record == harbor_record
    assert environment.answer == "public answer"
    assert (static_dir / "artifacts" / "answer.txt").read_text() == "public answer"
    assert "SECRET_" not in json.dumps(static_record)
