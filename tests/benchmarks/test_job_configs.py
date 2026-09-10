from pathlib import Path

import pytest
import yaml
from harbor.models.job.config import JobConfig


@pytest.mark.parametrize(
    ("name", "client"),
    [
        ("vllm", "vllm"),
        ("openai", "openai"),
        ("bedrock", "bedrock"),
        ("aliyun", "openai"),
    ],
)
def test_job_config(name: str, client: str) -> None:
    path = Path("configs/jobs/critpt") / f"{name}.yaml"
    config = JobConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    agent = config.agents[0]

    assert agent.name == ("ddsr_bench.benchmarks.critpt.evaluation.harbor:CritPtAgent")
    assert agent.kwargs["client_name"] == client
    assert agent.kwargs["sampling"]["max_tokens"] == 32768
    assert config.datasets[0].path == Path("tasks/critpt-official")
    assert config.jobs_dir == Path("outputs/harbor")


@pytest.mark.parametrize(
    ("name", "client"),
    [
        ("vllm", "vllm"),
        ("openai", "openai"),
        ("bedrock", "bedrock"),
        ("aliyun", "openai"),
    ],
)
def test_scicode_job_config(name: str, client: str) -> None:
    path = Path("configs/jobs/scicode") / f"{name}.yaml"
    config = JobConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    assert config.agents[0].name.endswith("scicode.evaluation.harbor:SciCodeAgent")
    assert config.agents[0].kwargs["client_name"] == client
    assert config.datasets[0].path == Path("tasks/scicode-validation")
    assert config.jobs_dir == Path("outputs/harbor")
