import json
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import sympy as sp

from ddsr_bench.benchmarks.critpt.evaluation.ai import docker, worker
from ddsr_bench.benchmarks.critpt.evaluation.ai.runtime import GraderTimeout
from ddsr_bench.benchmarks.critpt.evaluation.ai.validation import (
    UnsupportedAnswerOperation,
)
from ddsr_bench.benchmarks.critpt.evaluation.serialization import decode, encode
from ddsr_bench.benchmarks.utils import Resources


@pytest.mark.parametrize(
    "value", ["2*x", sp.Symbol("x", positive=True), {"A", "B"}, 1.25]
)
def test_values(monkeypatch, value):
    # Exercise the worker protocol without executing answer code on the host.
    monkeypatch.setattr(worker, "_execute_answer", lambda code: {"answer": value})
    result = worker.run("def answer():\n    return 2")
    assert result["status"] == "ok"
    assert decode(result["answer"]["answer"]) == value


@pytest.mark.parametrize(
    "mode,error",
    [
        ("ok", None),
        ("invalid", ValueError),
        ("unsupported", UnsupportedAnswerOperation),
        ("timeout", GraderTimeout),
        ("process", RuntimeError),
        ("protocol", RuntimeError),
        ("infrastructure", RuntimeError),
    ],
)
def test_container(tmp_path, monkeypatch, mode, error):
    calls = []
    code = "def answer():\n    return 2"

    class Container:
        async def upload_file(self, path, target):
            assert path.read_text() == code
            assert target == "/tmp/answer.py"
            calls.append("upload")

        async def exec(self, command, **kwargs):
            if command.startswith("python"):
                assert kwargs == {"timeout_sec": 10}
                if mode in {"timeout", "infrastructure"}:
                    raise RuntimeError(
                        "Command timed out after 10 seconds"
                        if mode == "timeout"
                        else "Docker unavailable"
                    )
                return SimpleNamespace(return_code=int(mode == "process"))
            data = (
                {"status": "ok", "answer": {"answer": encode(2)}}
                if mode == "ok"
                else {"status": mode, "error": "synthetic"}
            )
            return SimpleNamespace(return_code=0, stdout=json.dumps(data))

    @asynccontextmanager
    async def environment(directory, resources):
        try:
            yield Container()
        finally:
            calls.append("cleanup")

    monkeypatch.setattr(docker, "environment", environment)
    if error:
        with pytest.raises(error):
            docker.execute(code, resources=Resources("image", 1, 512, 10))
    else:
        assert docker.execute(code, resources=Resources("image", 1, 512, 10)) == {
            "answer": 2
        }
    assert calls == ["upload", "cleanup"]


@pytest.mark.skipif(
    not os.getenv("DDSR_TEST_AI_IMAGE"),
    reason="Set DDSR_TEST_AI_IMAGE for Docker smoke",
)
@pytest.mark.parametrize(
    "expression,expected",
    [("2", 2), ("1.25", 1.25), ("{'A', 'B'}", {"A", "B"}), ("sp.sqrt(2)", sp.sqrt(2))],
)
def test_live(expression, expected):
    # Only synthetic code executes, exclusively inside the Harbor container.
    resources = Resources(os.environ["DDSR_TEST_AI_IMAGE"], 1, 1024, 30)
    result = docker.execute(
        f"def answer():\n    return {expression}", resources=resources
    )
    assert result == {"answer": expected}
