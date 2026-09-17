"""Explicit path and namespace compatibility for migrated CritPt job records."""

from __future__ import annotations

from pathlib import Path

from harbor.models.job.config import JobConfig

LEGACY_AGENT = "critpt_eval.benchmark.harbor:CritPtAgent"
AGENT = "ddsr_bench.benchmarks.critpt.evaluation.harbor:CritPtAgent"


def check_resume(
    previous: JobConfig, current: JobConfig, migration: dict | None
) -> None:
    """Only concurrency, declared storage paths and equivalent client names may differ."""
    before = previous.model_dump(exclude={"n_concurrent_trials"})
    after = current.model_dump(exclude={"n_concurrent_trials"})
    if migration is not None:
        if not isinstance(migration, dict) or set(migration) != {"path_prefixes"}:
            raise ValueError("resume_migration requires only path_prefixes")
        prefixes = migration["path_prefixes"]
        if not isinstance(prefixes, dict) or not prefixes:
            raise ValueError("path_prefixes must be a nonempty mapping")
        for source, target in prefixes.items():
            if not isinstance(source, str) or not isinstance(target, str):
                raise TypeError("path prefixes must be strings")
            if not Path(target).is_absolute() or ".." in Path(source).parts:
                raise ValueError(
                    "mapped targets must be absolute; sources cannot contain .."
                )

        def mapped(path):
            if path is None:
                return None
            for source in sorted(prefixes, key=len, reverse=True):
                if Path(path).is_relative_to(source):
                    return Path(prefixes[source]) / Path(path).relative_to(source)
            return path

        before["jobs_dir"] = mapped(before["jobs_dir"])
        for dataset in before["datasets"]:
            dataset["path"] = mapped(dataset.get("path"))
        for agent in before["agents"]:
            if agent["name"] == LEGACY_AGENT:
                agent["name"] = AGENT
        # Both profiles send the same max_completion_tokens/reasoning/seed body.
        for old, new in zip(before["agents"], after["agents"], strict=True):
            old_kwargs, new_kwargs = old["kwargs"], new["kwargs"]
            if (
                old_kwargs.get("client_name") == "openai"
                and new_kwargs.get("client_name") == "aliyun"
                and new_kwargs.get("request_profile") == "openai"
                and old_kwargs.get("sampling", {}).get("enable_thinking") is None
                and old_kwargs.get("sampling", {}).get("top_k") is None
            ):
                old_kwargs.update(client_name="aliyun", request_profile="openai")
    if before != after:
        raise ValueError(
            "resume only permits changing n_concurrent_trials and explicit equivalent migration mappings"
        )
