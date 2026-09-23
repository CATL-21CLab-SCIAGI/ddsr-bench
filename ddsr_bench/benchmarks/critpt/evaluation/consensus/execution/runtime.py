"""Select worker execution; normal submission never falls back to the host."""

from __future__ import annotations

import asyncio
import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path

from ddsr_bench.benchmarks.utils import Resources
from ddsr_bench.grading import validation as code_validation

from .. import POLICY_VERSION
from ..references import checksum


def provenance(references_path: Path, runtime: Runtime) -> dict:
    return {
        "references_sha256": checksum(references_path.read_text()),
        "policy_version": POLICY_VERSION,
        "code_validation_sha256": checksum(Path(code_validation.__file__).read_text()),
        "execution": runtime.backend,
        "image": runtime.image if runtime.backend == "docker" else None,
        "image_id": runtime.image_id,
        **({"linux_sandbox": runtime.linux.info} if runtime.linux else {}),
        "host_python": platform.python_version(),
        "host_dependencies": {
            p: importlib.metadata.version(p) for p in ("sympy", "numpy", "scipy")
        },
    }


class Runtime:
    def __init__(
        self,
        *,
        backend: str = "docker",
        image: str = "ddsr-bench-critpt:latest",
        timeout: float = 60,
        cpus: int = 2,
        memory_mb: int = 4096,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if backend != "docker":
            raise ValueError("normal submission requires the docker backend")
        self.backend = backend
        if shutil.which("docker") is None:
            raise RuntimeError("Docker is required for internal submission")
        self.image, self.timeout = image, timeout
        self.cpus, self.memory_mb = cpus, memory_mb
        self.image_id = None
        self.linux = None
        if self.backend == "docker":
            try:
                inspected = subprocess.run(
                    ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.image_id = inspected.stdout.strip()
            except (subprocess.SubprocessError, OSError) as error:
                raise RuntimeError(
                    "Docker image unavailable; build the documented image first"
                ) from error

    def run(self, payload: dict) -> dict:
        if self.backend == "docker":
            from .docker import run

            resources = Resources(
                self.image_id, self.cpus, self.memory_mb, self.timeout
            )
            return asyncio.run(run(payload, resources))


class DiagnosticRuntime(Runtime):
    """Explicit Linux/host execution for reference diagnostics and reviewed tests."""

    def __init__(self, *, trusted_local=False, backend=None, timeout=60, **kwargs):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if backend not in (None, "docker", "linux"):
            raise ValueError("backend must be docker or linux")
        if trusted_local and backend is not None:
            raise ValueError("trusted_local and backend are mutually exclusive")
        if not trusted_local and backend != "linux":
            super().__init__(timeout=timeout, **kwargs)
            return
        self.backend = "trusted_local" if trusted_local else "linux"
        self.timeout = timeout
        self.image = kwargs.get("image", "ddsr-bench-critpt:latest")
        self.image_id = None
        self.linux = None
        if backend == "linux":
            from .linux import LinuxSandbox

            self.linux = LinuxSandbox(timeout)

    def run(self, payload):
        if self.backend == "docker":
            return super().run(payload)
        from .linux import run

        return run(
            payload,
            self.timeout,
            trusted_local=self.backend == "trusted_local",
            linux=self.linux,
        )
