"""Restricted Harbor-managed containers for CritPt answer workers."""

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from ddsr_bench.benchmarks.utils import Resources, write_json


@asynccontextmanager
async def environment(directory: Path, resources: Resources):
    """Start one worker without host mounts, references, or judge credentials.

    The caller transfers only its worker request. This creates an environment,
    not an agent trial; scoring and worker protocols belong to their callers.
    """
    directory.mkdir(parents=True, exist_ok=True)
    definition = directory / "environment"
    definition.mkdir()
    restrictions = directory / "restrictions.json"
    write_json(
        restrictions,
        {
            "services": {
                "main": {
                    "pull_policy": "never",
                    "network_mode": "none",
                    "init": True,
                    "read_only": True,
                    "cap_drop": ["ALL"],
                    "security_opt": ["no-new-privileges"],
                    "user": "65534:65534",
                    "pids_limit": 64,
                    "tmpfs": ["/tmp:rw,noexec,nosuid,size=64m"],
                    "logging": {"driver": "none"},
                    "ulimits": {"fsize": {"soft": 8_000_000, "hard": 8_000_000}},
                }
            }
        },
    )
    # Fixed network_mode:none avoids Harbor's redundant firewall sidecar.
    # init forwards shutdown signals and reaps children.
    container = DockerEnvironment(
        environment_dir=definition,
        environment_name="critpt-worker",
        session_id=f"critpt-worker-{uuid4().hex}",
        trial_paths=TrialPaths(directory / "trial"),
        task_env_config=EnvironmentConfig(
            docker_image=resources.image,
            cpus=resources.cpus,
            memory_mb=resources.memory_mb,
            workdir="/tmp",
        ),
        mounts=[],
        extra_docker_compose=[restrictions],
    )
    try:
        await container.start(force_build=False)
        yield container
    finally:
        await container.stop(delete=True)
