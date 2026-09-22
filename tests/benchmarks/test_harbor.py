from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "benchmark, name",
    [
        ("critpt", "CritPtAgent"),
        ("scicode", "SciCodeAgent"),
        ("cmphysbench", "CMPhysBenchAgent"),
        ("phybench", "PHYBenchAgent"),
    ],
)
@pytest.mark.parametrize(
    "client, key",
    [
        ("vllm", None),
        ("openai", "OPENAI_API_KEY"),
        ("bedrock", "AWS_BEARER_TOKEN_BEDROCK"),
        ("aliyun", "ALIYUN_API_KEY"),
    ],
)
def test_key_defaults(tmp_path, benchmark, name, client, key):
    module = import_module(f"ddsr_bench.benchmarks.{benchmark}.evaluation.harbor")
    agent_class = getattr(module, name)
    agent = agent_class(tmp_path, "teacher", client_name=client)
    assert agent.api_key_env == key
    custom = agent_class(
        tmp_path, "teacher", client_name=client, api_key_env="CUSTOM_KEY"
    )
    assert custom.api_key_env == "CUSTOM_KEY"
