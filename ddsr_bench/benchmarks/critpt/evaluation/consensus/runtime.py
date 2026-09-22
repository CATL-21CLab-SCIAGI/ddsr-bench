"""Bounded worker execution; Docker by default, Linux sandbox opt-in."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from ddsr_bench.grading import validation as code_validation

from . import POLICY_VERSION
from .bundle import digest


def provenance(bundle_path: Path, runtime: Runtime) -> dict:
    return {
        "bundle_sha256": digest(bundle_path.read_text()),
        "policy_version": POLICY_VERSION,
        "code_validation_sha256": digest(Path(code_validation.__file__).read_text()),
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
        trusted_local: bool = False,
        backend: str | None = None,
        image: str = "ddsr-critpt-consensus:local",
        timeout: float = 60,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if backend not in (None, "docker", "linux"):
            raise ValueError("backend must be docker or linux")
        if trusted_local and backend is not None:
            raise ValueError("trusted_local and backend are mutually exclusive")
        self.backend = "trusted_local" if trusted_local else backend or "docker"
        if self.backend == "docker" and shutil.which("docker") is None:
            raise RuntimeError(
                "Docker is required. Alternatively select --backend linux with bubblewrap. "
                "Use --trusted-local only for code you have reviewed and trust."
            )
        self.trusted_local, self.image, self.timeout = trusted_local, image, timeout
        self.image_id = None
        self.linux = None
        if self.backend == "linux":
            from .linux import LinuxSandbox

            self.linux = LinuxSandbox(timeout)
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
        name = "critpt-consensus-" + uuid.uuid4().hex
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[5]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "CONSENSUS_CPU_SECONDS": str(int(self.timeout) + 2),
        }
        if self.trusted_local:
            command = [
                sys.executable,
                "-m",
                "ddsr_bench.benchmarks.critpt.evaluation.consensus.worker",
            ]
        elif self.linux is not None:
            command = self.linux.command()
            env = self.linux.env
        else:
            command = [
                "docker",
                "run",
                "--rm",
                "-i",
                "--name",
                name,
                "--pull=never",
                "--log-driver=none",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--user=65534:65534",
                "--pids-limit=64",
                "--memory=1g",
                "--cpus=1",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
                "--workdir=/tmp",
                "-e",
                "OPENBLAS_NUM_THREADS=1",
                "-e",
                "OMP_NUM_THREADS=1",
                "-e",
                f"CONSENSUS_CPU_SECONDS={int(self.timeout) + 2}",
                self.image_id,
                "python",
                "-m",
                "ddsr_bench.benchmarks.critpt.evaluation.consensus.worker",
            ]
        with (
            tempfile.TemporaryDirectory(prefix="critpt-consensus-") as cwd,
            tempfile.TemporaryFile() as request,
            tempfile.TemporaryFile() as stdout,
            tempfile.TemporaryFile() as stderr,
        ):
            # A regular stdin file avoids partial pipe writes when polling a slow
            # worker. Docker still forwards these bytes through its stdin only.
            encoded = json.dumps(payload, allow_nan=False).encode()
            if len(encoded) > 8_000_000:
                return {
                    "status": "error",
                    "stage": "input_limit",
                    "error": "worker request too large",
                }
            request.write(encoded)
            request.seek(0)
            process = subprocess.Popen(
                command,
                stdin=request,
                stdout=stdout,
                stderr=stderr,
                cwd=cwd,
                env=env,
                start_new_session=True,
            )
            started = time.monotonic()
            failure = None
            try:
                while True:
                    try:
                        process.wait(timeout=min(0.1, self.timeout))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() - started >= self.timeout:
                            failure = "timeout"
                        if (
                            max(
                                os.fstat(stdout.fileno()).st_size,
                                os.fstat(stderr.fileno()).st_size,
                            )
                            > 8_000_000
                        ):
                            failure = "output_limit"
                        if failure:
                            break
            except BaseException:
                self._stop(process, name)
                raise
            if failure:
                self._stop(process, name)
                return {
                    "status": "error",
                    "stage": failure,
                    "error": "worker resource limit exceeded",
                }
            stdout.seek(0)
            raw = stdout.read(8_000_001)
            if process.returncode or len(raw) > 8_000_000:
                return {
                    "status": "error",
                    "stage": "process",
                    "error": f"worker exited {process.returncode} or output limit exceeded",
                }
            try:
                data = json.loads(raw)
                if not isinstance(data, dict) or data.get("status") not in {
                    "ok",
                    "error",
                }:
                    raise ValueError("invalid result")
                return data
            except (ValueError, UnicodeDecodeError):
                return {
                    "status": "error",
                    "stage": "protocol",
                    "error": "invalid worker response",
                }

    def _stop(self, process, name):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        if self.backend == "docker":
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                timeout=10,
                check=False,
            )
