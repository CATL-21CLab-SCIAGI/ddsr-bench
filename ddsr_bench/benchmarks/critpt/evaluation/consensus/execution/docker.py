"""Docker execution backend using Harbor-managed worker environments."""

import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from ddsr_bench.benchmarks.utils import Resources, write_json


@asynccontextmanager
async def environment(directory: Path, resources: Resources):
    """Start a fresh worker environment and clean up even after startup failure.

    No host directories are mounted: an answer must never see references or
    another worker's files. The caller transfers only that worker's request.
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
    # Docker's fixed network_mode:none enforces isolation for this sole service.
    # Do not request Harbor's dynamic firewall: it adds a redundant sidecar.
    # init forwards shutdown signals and reaps children without changing timeouts.
    container = DockerEnvironment(
        environment_dir=definition,
        environment_name="critpt-consensus",
        session_id=f"critpt-consensus-{uuid4().hex}",
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


async def run(payload: dict, resources: Resources) -> dict:
    """Exchange one bounded request with the unchanged consensus worker."""
    encoded = json.dumps(payload, allow_nan=False).encode()
    if len(encoded) > 8_000_000:
        return {
            "status": "error",
            "stage": "input_limit",
            "error": "worker request too large",
        }
    with TemporaryDirectory(prefix="critpt-consensus-") as temporary:
        directory = Path(temporary)
        request = directory / "request.json"
        # Preserve the exact bounded wire encoding (not pretty-printed JSON).
        request.write_bytes(encoded)
        request.chmod(0o644)
        async with environment(directory, resources) as container:
            await container.upload_file(request, "/tmp/request.json")
            try:
                result = await container.exec(
                    "python -m ddsr_bench.benchmarks.critpt.evaluation.consensus.execution.worker "
                    "< /tmp/request.json > /tmp/response.json 2> /tmp/worker.err",
                    env={
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "CONSENSUS_CPU_SECONDS": str(int(resources.timeout_sec) + 2),
                    },
                    timeout_sec=math.ceil(resources.timeout_sec),
                )
            except (TimeoutError, RuntimeError) as error:
                # Harbor 0.22 wraps its exec timeout in RuntimeError. Do not
                # misreport unrelated Docker/infrastructure errors as timeouts.
                if isinstance(error, RuntimeError) and str(error) != (
                    f"Command timed out after {math.ceil(resources.timeout_sec)} seconds"
                ):
                    raise
                return {
                    "status": "error",
                    "stage": "timeout",
                    "error": "worker resource limit exceeded",
                }
            if result.return_code:
                return {
                    "status": "error",
                    "stage": "process",
                    "error": f"worker exited {result.return_code} or output limit exceeded",
                }
            output = await container.exec(
                "head -c 8000001 /tmp/response.json", timeout_sec=10
            )
            try:
                raw = output.stdout or ""
                if output.return_code or len(raw.encode()) > 8_000_000:
                    raise ValueError("invalid output")
                data = json.loads(raw)
                if not isinstance(data, dict) or data.get("status") not in {
                    "ok",
                    "error",
                }:
                    raise ValueError("invalid result")
                return data
            except ValueError:
                return {
                    "status": "error",
                    "stage": "protocol",
                    "error": "invalid worker response",
                }
