from copy import deepcopy

import pytest
from harbor.models.job.config import JobConfig

from ddsr_bench.benchmarks.critpt.evaluation.resume import (
    AGENT,
    LEGACY_AGENT,
    check_resume,
)


def configs():
    previous = {
        "job_name": "legacy",
        "jobs_dir": "/old/outputs/harbor",
        "agents": [
            {
                "name": LEGACY_AGENT,
                "model_name": "solver",
                "kwargs": {
                    "client_name": "openai",
                    "style": "two-step",
                    "seed_base": 42,
                    "base_url": "https://example.com/v1",
                    "formatting_max_tokens": 65536,
                    "sampling": {"max_tokens": 393216, "reasoning_effort": "max"},
                },
            }
        ],
        "datasets": [{"path": "/old/tasks"}],
    }
    current = deepcopy(previous)
    current["jobs_dir"] = "/assets/outputs/harbor"
    current["datasets"][0]["path"] = "/assets/tasks"
    current["agents"][0]["name"] = AGENT
    current["agents"][0]["kwargs"].update(
        client_name="aliyun", request_profile="openai"
    )
    current["n_concurrent_trials"] = 32
    return previous, current


def test_migrated_resume_requires_explicit_path_map_and_equivalent_profile():
    before, after = configs()
    previous, current = JobConfig.model_validate(before), JobConfig.model_validate(
        after
    )
    with pytest.raises(ValueError, match="only permits"):
        check_resume(previous, current, None)
    check_resume(previous, current, {"path_prefixes": {"/old": "/assets"}})


@pytest.mark.parametrize(
    "key,value",
    [
        ("style", "one-step"),
        ("seed_base", 43),
        ("formatting_max_tokens", 131072),
        ("request_profile", "native"),
        ("base_url", "https://different.example.com/v1"),
        ("sampling", {"max_tokens": 393216, "reasoning_effort": "high"}),
    ],
)
def test_migration_does_not_relax_generation_identity(key, value):
    before, after = configs()
    after["agents"][0]["kwargs"][key] = value
    with pytest.raises(ValueError, match="only permits"):
        check_resume(
            JobConfig.model_validate(before),
            JobConfig.model_validate(after),
            {"path_prefixes": {"/old": "/assets"}},
        )
