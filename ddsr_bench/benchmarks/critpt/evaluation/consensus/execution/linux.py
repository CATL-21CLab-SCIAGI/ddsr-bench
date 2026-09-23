"""Linux diagnostic backend: Bubblewrap isolation and subprocess supervision.

Selected explicitly through the historical CLI with --backend linux, never as an
automatic fallback. Requires a compatible Linux host; see consensus/README.md.
The explicit trusted-local diagnostic option reuses supervision without
Bubblewrap; it is not a separate backend module or a safe execution sandbox.

Only the interpreter, its standard library, four scientific packages and the
worker's source modules are exposed. Never bind the repository or a site-packages
directory wholesale. The trusted bootstrap applies hard limits and seccomp
before importing the worker or reading a answer request.
"""

from __future__ import annotations

import ctypes
import errno
import importlib.metadata
import importlib.util
import itertools
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

PACKAGES = ("numpy", "scipy", "sympy", "mpmath")
MODULES = (
    "__init__",
    "execution/__init__",
    "execution/worker",
    "execution/serialization",
    "execution/linux",
    "matching/__init__",
    "matching/compare",
    "matching/cases",
    "matching/rules",
)
LIMITS = {"address_space": 1 << 30, "file_bytes": 8_000_000, "tmp_bytes": 64 << 20}


class LinuxSandbox:
    def __init__(self, timeout: float):
        if sys.platform != "linux":
            raise RuntimeError("The linux backend requires Linux")
        self.bwrap = shutil.which("bwrap")
        if not self.bwrap:
            raise RuntimeError("The linux backend requires bubblewrap (bwrap)")
        self.timeout = timeout
        self.cpus = itertools.cycle(sorted(os.sched_getaffinity(0)))
        self.mounts: dict[str, Path] = {}
        binary = Path(sys._base_executable).resolve()
        self.bind(binary, "/runtime/bin/python")
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        self.stdlib = f"/runtime/lib/{version}"
        self.bind(Path(sysconfig.get_path("stdlib")), self.stdlib)
        # Loaders and native extensions use distribution-specific library paths.
        multiarch = sysconfig.get_config_var("MULTIARCH")
        if multiarch:
            for prefix in ("/lib", "/usr/lib"):
                path = Path(prefix) / multiarch
                if path.is_dir():
                    self.bind(path, str(path))
        try:
            linked = subprocess.run(
                ["ldd", str(binary)],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                "Linux sandbox requires ldd and a dynamically linked CPython"
            ) from error
        for word in linked.stdout.split():
            if word.startswith("/") and Path(word).is_file():
                self.bind(Path(word), word)
        for name in PACKAGES:
            spec = importlib.util.find_spec(name)
            if spec is None or not spec.origin:
                raise RuntimeError(f"Missing linux worker dependency: {name}")
            package = Path(spec.origin).resolve().parent
            self.bind(package, f"/runtime/site/{name}")
            libraries = package.parent / f"{name}.libs"
            if libraries.is_dir():
                self.bind(libraries, f"/runtime/site/{name}.libs")
            dist = importlib.metadata.distribution(name)
            metadata = next(
                (f for f in dist.files or [] if str(f).endswith(".dist-info/METADATA")),
                None,
            )
            if metadata is None:
                raise RuntimeError(f"Missing linux worker package metadata: {name}")
            self.bind(Path(dist.locate_file(metadata)), f"/runtime/site/{metadata}")
        package = Path(__file__).resolve().parent.parent
        root = package.parents[3]
        # Bind namespace initializers individually, never the package trees.
        for parent in (root, *reversed(package.parents[:3])):
            relative = parent.relative_to(root.parent)
            self.bind(parent / "__init__.py", f"/app/{relative}/__init__.py")
        self.bind(root / "grading/__init__.py", "/app/ddsr_bench/grading/__init__.py")
        self.bind(
            root / "grading/validation.py", "/app/ddsr_bench/grading/validation.py"
        )
        for module in MODULES:
            self.bind(
                package / f"{module}.py",
                f"/app/ddsr_bench/benchmarks/critpt/evaluation/consensus/{module}.py",
            )
        self.env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"}
        try:
            probe = subprocess.run(
                self.command(probe=True),
                env=self.env,
                capture_output=True,
                text=True,
                timeout=15,
                start_new_session=True,
                check=False,
            )
            if probe.returncode:
                raise RuntimeError(probe.stderr[-1500:])
            self.info = json.loads(probe.stdout)
            self.info["bubblewrap"] = subprocess.check_output(
                [self.bwrap, "--version"], text=True, env=self.env, timeout=5
            ).strip()
        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
            raise RuntimeError(f"Linux sandbox preflight failed: {error}") from error

    def bind(self, source: Path, destination: str):
        self.mounts[destination] = source.resolve(strict=True)

    def command(self, *, probe: bool = False) -> list[str]:
        args = [
            self.bwrap,
            "--unshare-all",
            "--unshare-user",
            "--uid",
            "65534",
            "--gid",
            "65534",
            "--cap-drop",
            "ALL",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
        ]
        for destination, source in self.mounts.items():
            args += ["--ro-bind", str(source), destination]
        # Some Python installations nest site-packages under the stdlib.
        for name in ("site-packages", "dist-packages"):
            if not (self.mounts[self.stdlib] / name).exists():
                continue
            args += [
                "--tmpfs",
                f"{self.stdlib}/{name}",
                "--remount-ro",
                f"{self.stdlib}/{name}",
            ]
        # The numerical worker needs no procfs (some container hosts forbid
        # mounting it). Leaving it absent also hides process file descriptors.
        args += ["--dir", "/dev"]
        for name in ("null", "zero", "urandom", "random"):
            args += ["--dev-bind", f"/dev/{name}", f"/dev/{name}"]
        args += [
            "--size",
            str(LIMITS["tmp_bytes"]),
            "--tmpfs",
            "/tmp",
            "--chmod",
            "1777",
            "/tmp",
            "--remount-ro",
            "/",
            "--chdir",
            "/tmp",
        ]
        env = {
            "PYTHONHOME": "/runtime",
            "PYTHONPATH": "/app:/runtime/site",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "LC_ALL": "C.UTF-8",
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "CONSENSUS_CPU_SECONDS": str(int(self.timeout) + 2),
            "CONSENSUS_CPU": str(next(self.cpus)),
        }
        for key, value in env.items():
            args += ["--setenv", key, value]
        args += [
            "/runtime/bin/python",
            "-S",
            "-P",
            "-m",
            "ddsr_bench.benchmarks.critpt.evaluation.consensus.execution.linux",
        ]
        if probe:
            args.append("--probe")
        return args


def run(payload: dict, timeout: float, *, trusted_local: bool, linux) -> dict:
    """Run one diagnostic worker; trusted_local explicitly bypasses Bubblewrap."""
    if not trusted_local and linux is None:
        raise ValueError("legacy execution requires an explicit backend")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[6]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "CONSENSUS_CPU_SECONDS": str(int(timeout) + 2),
    }
    if trusted_local:
        command = [
            sys.executable,
            "-m",
            "ddsr_bench.benchmarks.critpt.evaluation.consensus.execution.worker",
        ]
    elif linux is not None:
        command = linux.command()
        env = linux.env
    with (
        tempfile.TemporaryDirectory(prefix="critpt-consensus-") as cwd,
        tempfile.TemporaryFile() as request,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        # A regular stdin file avoids partial pipe writes when polling a slow
        # worker in the legacy Linux/local paths.
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
                    process.wait(timeout=min(0.1, timeout))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started >= timeout:
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
            _stop(process)
            raise
        if failure:
            _stop(process)
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


def _stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def harden():
    """Apply irreversible limits before any model-controlled Python runs."""
    os.umask(0o077)
    os.sched_setaffinity(0, {int(os.environ["CONSENSUS_CPU"])})
    for which, limit in (
        (resource.RLIMIT_AS, LIMITS["address_space"]),
        (resource.RLIMIT_FSIZE, LIMITS["file_bytes"]),
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(which, (limit, limit))
    seconds = int(os.environ["CONSENSUS_CPU_SECONDS"])
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
    # RLIMIT_NPROC is UID-wide and can be ineffective for a host-root caller.
    # Block process/thread creation instead; numerical libraries use one thread.
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError("seccomp initialization failed")
    denied = (
        "clone",
        "clone3",
        "fork",
        "vfork",
        "execve",
        "execveat",
        "unshare",
        "setns",
        "mount",
        "umount2",
        "pivot_root",
        "chroot",
        "ptrace",
        "process_vm_readv",
        "process_vm_writev",
        "bpf",
        "perf_event_open",
        "userfaultfd",
        "io_uring_setup",
        "keyctl",
        "add_key",
        "request_key",
        "open_by_handle_at",
        "socket",
        "socketpair",
        "sched_setaffinity",
        "kill",
        "tkill",
        "tgkill",
        "setsid",
        "setpgid",
    )
    try:
        for name in denied:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number == -1:
                raise RuntimeError(f"libseccomp does not recognize {name}")
            if lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, number, 0):
                raise RuntimeError(f"Cannot block syscall {name}")
        if lib.seccomp_load(ctx):
            raise RuntimeError("Cannot load seccomp policy")
    finally:
        lib.seccomp_release(ctx)


def main():
    harden()
    if "--probe" in sys.argv:
        # Import the full dependency set inside the real sandbox and limits.
        import scipy  # noqa: F401

        from . import worker  # noqa: F401

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4

        class Header(ctypes.Structure):
            _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

        class Caps(ctypes.Structure):
            _fields_ = [
                ("effective", ctypes.c_uint32),
                ("permitted", ctypes.c_uint32),
                ("inheritable", ctypes.c_uint32),
            ]

        header, caps = Header(0x20080522, 0), (Caps * 2)()
        privilege_info = {
            "uid": os.getuid(),
            "gid": os.getgid(),
            "no_new_privs": libc.prctl(39, 0, 0, 0, 0),
            "seccomp": libc.prctl(21, 0, 0, 0, 0),
            "capget": libc.capget(ctypes.byref(header), ctypes.byref(caps)),
            "caps": [(c.effective, c.permitted, c.inheritable) for c in caps],
        }
        if (
            privilege_info["uid"] != 65534
            or privilege_info["gid"] != 65534
            or privilege_info["no_new_privs"] != 1
            or privilege_info["seccomp"] != 2
            or privilege_info["capget"] != 0
            or any(c.effective or c.permitted or c.inheritable for c in caps)
        ):
            raise RuntimeError(f"Worker privileges were not removed: {privilege_info}")
        print(
            json.dumps(
                {
                    "backend": "linux",
                    "policy": "consensus-linux-v1",
                    "limits": LIMITS,
                    "python": sys.version.split()[0],
                    "dependencies": {
                        p: importlib.metadata.version(p) for p in PACKAGES
                    },
                    "process_creation": "denied",
                    "network": "denied",
                }
            )
        )
    else:
        from .worker import main as worker_main

        worker_main()


if __name__ == "__main__":
    main()
