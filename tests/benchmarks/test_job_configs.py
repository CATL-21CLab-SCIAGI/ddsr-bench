from pathlib import Path

import pytest
import yaml
from harbor.models.job.config import JobConfig


@pytest.mark.parametrize(
    ("name", "client", "tokens"),
    [("deepseek-pai-max", "aliyun", 393216), ("qwen-long-context", "vllm", 262144)],
)
def test_long_context_config(name: str, client: str, tokens: int) -> None:
    path = Path("configs/jobs/critpt") / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = JobConfig.model_validate(raw)
    kwargs = config.agents[0].kwargs
    assert raw["benchmark"] == "critpt"
    assert config.n_attempts == 5
    assert kwargs["client_name"] == client
    assert kwargs["sampling"]["max_tokens"] == tokens
    assert kwargs["formatting_max_tokens"] == 131072
    if client == "aliyun":
        assert kwargs["api_key_env"] == "PAI_API_KEY"
        assert kwargs["request_profile"] == "openai"
    else:
        assert kwargs["context_window"] == 262144


@pytest.mark.parametrize(
    ("name", "client"),
    [
        ("vllm", "vllm"),
        ("openai", "openai"),
        ("bedrock", "bedrock"),
        ("aliyun", "aliyun"),
    ],
)
def test_job_config(name: str, client: str) -> None:
    path = Path("configs/jobs/critpt") / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = JobConfig.model_validate(raw)
    agent = config.agents[0]

    assert raw["benchmark"] == "critpt"
    assert agent.name == ("ddsr_bench.benchmarks.critpt.evaluation.harbor:CritPtAgent")
    assert agent.kwargs["client_name"] == client
    assert agent.kwargs["sampling"]["max_tokens"] == 32768
    assert config.datasets[0].path == Path("tasks/critpt-official")
    assert config.jobs_dir == Path("outputs/harbor")
    suffix = "" if name == "vllm" else f"-{name}"
    assert config.job_name == f"critpt-official{suffix}"


@pytest.mark.parametrize(
    ("name", "client"),
    [
        ("vllm", "vllm"),
        ("openai", "openai"),
        ("bedrock", "bedrock"),
        ("aliyun", "aliyun"),
    ],
)
def test_scicode_job_config(name: str, client: str) -> None:
    path = Path("configs/jobs/scicode") / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = JobConfig.model_validate(raw)

    assert raw["benchmark"] == "scicode"
    assert config.agents[0].name.endswith("scicode.evaluation.harbor:SciCodeAgent")
    assert config.agents[0].kwargs["client_name"] == client
    assert config.datasets[0].path == Path("tasks/scicode-validation")
    assert config.jobs_dir == Path("outputs/harbor")
    suffix = "" if name == "vllm" else f"-{name}"
    assert config.job_name == f"scicode-validation{suffix}"


@pytest.mark.parametrize(
    ("name", "client"),
    [
        ("vllm", "vllm"),
        ("openai", "openai"),
        ("bedrock", "bedrock"),
        ("aliyun", "aliyun"),
    ],
)
def test_cmphysbench_job_config(name: str, client: str) -> None:
    path = Path("configs/jobs/cmphysbench") / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = JobConfig.model_validate(raw)

    assert raw["benchmark"] == "cmphysbench"
    assert config.agents[0].name.endswith(
        "cmphysbench.evaluation.harbor:CMPhysBenchAgent"
    )
    assert config.agents[0].kwargs["client_name"] == client
    assert config.agents[0].kwargs["sampling"]["max_tokens"] == 16384
    assert config.datasets[0].path == Path("tasks/cmphysbench")
    assert config.jobs_dir == Path("outputs/harbor")
    assert config.verifier.override_timeout_sec is None
    suffix = "" if name == "vllm" else f"-{name}"
    assert config.job_name == f"cmphysbench-train{suffix}"
    if name in ("vllm", "aliyun"):
        assert config.agents[0].kwargs["sampling"]["temperature"] == 0.6
        assert config.agents[0].kwargs["sampling"]["top_p"] == 0.95


@pytest.mark.parametrize(
    ("name", "client"),
    [
        ("vllm", "vllm"),
        ("openai", "openai"),
        ("bedrock", "bedrock"),
        ("aliyun", "aliyun"),
    ],
)
def test_phybench_job_config(name: str, client: str) -> None:
    path = Path("configs/jobs/phybench") / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = JobConfig.model_validate(raw)

    assert raw["benchmark"] == "phybench"
    assert config.agents[0].name.endswith("phybench.evaluation.harbor:PHYBenchAgent")
    assert config.agents[0].kwargs["client_name"] == client
    assert config.agents[0].kwargs["sampling"]["max_tokens"] == 32768
    assert config.datasets[0].path == Path("tasks/phybench")
    assert config.jobs_dir == Path("outputs/harbor")
    suffix = "" if name == "vllm" else f"-{name}"
    assert config.job_name == f"phybench-gradable{suffix}"
    if name in ("vllm", "aliyun"):
        assert config.agents[0].kwargs["sampling"]["temperature"] == 0.6
        assert config.agents[0].kwargs["sampling"]["top_p"] == 0.95
