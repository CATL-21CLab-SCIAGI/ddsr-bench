"""Real OS boundary tests, deliberately independent of the AST validator."""

import errno
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus import linux
from ddsr_bench.benchmarks.critpt.evaluation.consensus.bundle import DEFAULT_BUNDLE
from ddsr_bench.benchmarks.critpt.evaluation.consensus.cli import parser, provenance
from ddsr_bench.benchmarks.critpt.evaluation.consensus.runtime import Runtime


@pytest.fixture(scope="module")
def sandbox():
    if sys.platform != "linux" or not shutil.which("bwrap"):
        pytest.skip("Linux isolation integration tests require Linux and bubblewrap")
    # If installed but misconfigured, fail: never silently test on the host.
    return Runtime(backend="linux", timeout=10)


def inside(sandbox, code):
    command = sandbox.linux.command()
    command = command[: command.index("-m")] + [
        "-c",
        "from ddsr_bench.benchmarks.critpt.evaluation.consensus.linux import harden; harden()\n"
        + code,
    ]
    result = subprocess.run(
        command,
        env=sandbox.linux.env,
        capture_output=True,
        text=True,
        timeout=15,
        start_new_session=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def evaluate(runtime, body):
    return runtime.run(
        {
            "action": "evaluate",
            "code": "def answer():\n    " + body,
            "template": "def answer(): pass",
            "inputs": [{}],
        }
    )


def test_missing_bubblewrap_fails_closed(monkeypatch):
    monkeypatch.setattr(linux.sys, "platform", "linux")
    monkeypatch.setattr(linux.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="requires bubblewrap"):
        Runtime(backend="linux")


def test_backend_selection_is_explicit():
    with pytest.raises(ValueError, match="mutually exclusive"):
        Runtime(backend="linux", trusted_local=True)
    with pytest.raises(ValueError, match="backend must"):
        Runtime(backend="auto")
    with pytest.raises(SystemExit):
        parser().parse_args(
            ["replay", "--output", "unused", "--backend", "linux", "--trusted-local"]
        )


def test_failed_preflight_fails_closed(sandbox, monkeypatch):
    def run(command, **kwargs):
        if command[0] == "ldd":
            return subprocess.CompletedProcess(command, 0, stdout="")
        return subprocess.CompletedProcess(command, 1, stderr="namespace denied")

    monkeypatch.setattr(linux.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="preflight failed.*namespace denied"):
        Runtime(backend="linux")


def test_worker_protocol_and_provenance(sandbox, sample_bundle_path):
    result = evaluate(sandbox, "print('discard me'); return 42")
    assert result["status"] == "ok"
    assert result["outputs"] == [{"t": "int", "v": "42"}]
    metadata = provenance(sample_bundle_path, sandbox)
    assert metadata["execution"] == "linux"
    assert metadata["image_id"] is None
    assert metadata["linux_sandbox"]["policy"] == "consensus-linux-v1"
    assert metadata["linux_sandbox"]["dependencies"]["numpy"]


def test_host_files_credentials_and_reference_bundle_are_hidden(
    sandbox, tmp_path, monkeypatch
):
    secret = tmp_path / "host-secret"
    secret.write_text("test secret, not a real credential")
    monkeypatch.setenv("CONSENSUS_TEST_SECRET", "must not be inherited")
    paths = [
        str(secret),
        str(DEFAULT_BUNDLE),
        str(Path(__file__).resolve().parents[4] / ".env"),
        "/proc/self/environ",
        "/proc/self/fd",
        "/root",
        "/sys",
    ]
    result = inside(
        sandbox,
        f"""
import json, os
paths = {paths!r}
assert 'CONSENSUS_TEST_SECRET' not in os.environ
assert not any('KEY' in key or 'TOKEN' in key or 'PROXY' in key for key in os.environ)
for path in paths:
    assert not os.path.exists(path), path
for path in ['/forbidden', '/app/forbidden']:
    try:
        open(path, 'w')
        raise AssertionError('writable root')
    except OSError:
        pass
open('/tmp/private', 'w').write('temporary')
assert os.stat('/tmp/private').st_mode & 0o777 == 0o600
print(json.dumps(True))
""",
    )
    assert result is True
    assert secret.read_text() == "test secret, not a real credential"
    assert inside(
        sandbox,
        "import json, os; print(json.dumps(not os.path.exists('/tmp/private')))",
    )


@pytest.mark.parametrize(
    "operation",
    [
        "os.fork()",
        "os.execv('/runtime/bin/python', ['python', '-c', 'pass'])",
        "socket.socket()",
        "socket.socketpair()",
        "os.sched_setaffinity(0, os.sched_getaffinity(0))",
        "os.kill(os.getpid(), 0)",
        "os.unshare(os.CLONE_NEWUSER)",
    ],
)
def test_syscalls_denied_even_without_validation(sandbox, operation):
    result = inside(
        sandbox,
        f"""
import os, socket, json
try:
    {operation}
except OSError as error:
    print(json.dumps(error.errno))
else:
    raise AssertionError('operation was allowed')
""",
    )
    assert result == errno.EPERM


def test_limits_cannot_be_raised_and_large_allocation_fails(sandbox):
    assert inside(
        sandbox,
        """
import os, resource, json
assert len(os.sched_getaffinity(0)) == 1
for limit in (resource.RLIMIT_AS, resource.RLIMIT_FSIZE, resource.RLIMIT_NOFILE):
    soft, hard = resource.getrlimit(limit)
    assert soft == hard
    try:
        resource.setrlimit(limit, (hard + 1, hard + 1))
        raise AssertionError('raised hard resource limit')
    except ValueError:
        pass
try:
    bytearray(2 << 30)
    raise AssertionError('allocation exceeded limit')
except MemoryError:
    pass
print(json.dumps(True))
""",
    )


def test_files_and_tmpfs_are_bounded(sandbox):
    assert inside(
        sandbox,
        """
import errno, json, os
try:
    with open('/tmp/large', 'wb') as f:
        f.write(b'x' * 8_000_001)
    raise AssertionError('file size exceeded limit')
except OSError as error:
    assert error.errno == errno.EFBIG
os.unlink('/tmp/large')
try:
    for i in range(10):
        with open('/tmp/' + str(i), 'wb') as f:
            f.write(b'x' * 8_000_000)
    raise AssertionError('tmpfs size exceeded limit')
except OSError as error:
    assert error.errno == errno.ENOSPC
print(json.dumps(True))
""",
    )


def test_worker_timeout_and_cleanup(sandbox, monkeypatch):
    monkeypatch.setattr(sandbox, "timeout", 1)
    assert evaluate(sandbox, "while True: pass")["stage"] == "timeout"
    assert evaluate(sandbox, "return 7")["status"] == "ok"


def test_interrupt_reaps_worker(sandbox, monkeypatch):
    original = subprocess.Popen.wait
    interrupted = []

    def wait(process, *args, **kwargs):
        if not interrupted:
            interrupted.append(process)
            raise KeyboardInterrupt
        return original(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    with pytest.raises(KeyboardInterrupt):
        evaluate(sandbox, "while True: pass")
    assert interrupted[0].returncode is not None
    assert not Path(f"/proc/{interrupted[0].pid}").exists()
