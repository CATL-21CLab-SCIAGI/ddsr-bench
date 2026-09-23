import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus.execution import docker
from ddsr_bench.benchmarks.utils import Resources


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "start", "worker"])
async def test_lifecycle(tmp_path, monkeypatch, failure):
    calls = []
    settings = {}

    class FakeEnvironment:
        def __init__(self, **kwargs):
            settings.update(kwargs)

        async def start(self, *, force_build):
            calls.append(("start", force_build))
            if failure == "start":
                raise RuntimeError("startup failure")

        async def stop(self, *, delete):
            calls.append(("stop", delete))

    monkeypatch.setattr(docker, "DockerEnvironment", FakeEnvironment)
    resources = Resources("ddsr-bench-critpt:latest", 2, 4096, 120)

    async def run():
        async with docker.environment(tmp_path, resources):
            if failure == "worker":
                raise RuntimeError("worker failure")

    if failure:
        with pytest.raises(RuntimeError, match="failure"):
            await run()
    else:
        await run()
    assert calls == [("start", False), ("stop", True)]
    assert settings["mounts"] == []
    config = settings["task_env_config"]
    assert (config.docker_image, config.cpus, config.memory_mb) == (
        resources.image,
        resources.cpus,
        resources.memory_mb,
    )
    assert "network_policy" not in settings  # Docker enforces fixed isolation.
    restrictions = json.loads(settings["extra_docker_compose"][0].read_text())
    service = restrictions["services"]["main"]
    assert service["read_only"] and service["network_mode"] == "none"
    assert service["init"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["user"] == "65534:65534"
    assert service["security_opt"] == ["no-new-privileges"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ok", "timeout", "process", "protocol"])
async def test_worker(tmp_path, monkeypatch, mode):
    calls = []
    payload = {"action": "evaluate", "code": "private request"}

    class Worker:
        async def upload_file(self, source, target):
            assert json.loads(source.read_bytes()) == payload
            assert target == "/tmp/request.json"

        async def exec(self, command, **kwargs):
            calls.append(command)
            assert "private request" not in command
            if len(calls) == 1:
                if mode == "timeout":
                    raise RuntimeError("Command timed out after 10 seconds")
                return SimpleNamespace(return_code=1 if mode == "process" else 0)
            return SimpleNamespace(
                return_code=0, stdout='{"status":"ok"}' if mode == "ok" else "invalid"
            )

    @asynccontextmanager
    async def environment(directory, resources):
        try:
            yield Worker()
        finally:
            calls.append("cleanup")

    monkeypatch.setattr(docker, "environment", environment)
    result = await docker.run(payload, Resources("image", 2, 4096, 10))
    assert result["status"] == ("ok" if mode == "ok" else "error")
    if mode != "ok":
        assert result["stage"] == mode
    assert calls[-1] == "cleanup"
