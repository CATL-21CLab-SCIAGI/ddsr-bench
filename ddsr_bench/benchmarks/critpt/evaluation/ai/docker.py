"""Run validated AI answer code in a fresh Harbor-managed container."""

import asyncio
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from ddsr_bench.benchmarks.utils import Resources

from ..docker import environment
from ..serialization import decode
from .runtime import GraderTimeout
from .validation import UnsupportedAnswerOperation


def execute(code: str, *, resources: Resources) -> dict:
    """Synchronous grade_response callback; no host execution fallback."""
    return asyncio.run(_execute(code, resources))


async def _execute(code: str, resources: Resources) -> dict:
    encoded = code.encode("utf-8")
    if len(encoded) > 8_000_000:
        raise ValueError("answer code exceeds worker input limit")
    timeout = math.ceil(resources.timeout_sec)
    with TemporaryDirectory(prefix="critpt-ai-") as temporary:
        directory = Path(temporary)
        path = directory / "answer.py"
        path.write_bytes(encoded)
        path.chmod(0o644)
        async with environment(directory, resources) as container:
            await container.upload_file(path, "/tmp/answer.py")
            try:
                result = await container.exec(
                    "python -m ddsr_bench.benchmarks.critpt.evaluation.ai.worker "
                    "< /tmp/answer.py > /tmp/result.json 2> /tmp/worker.err",
                    timeout_sec=timeout,
                )
            except (TimeoutError, RuntimeError) as error:
                if isinstance(error, RuntimeError) and str(error) != (
                    f"Command timed out after {timeout} seconds"
                ):
                    raise
                raise GraderTimeout from error
            if result.return_code:
                raise RuntimeError("AI answer worker exited without a result")
            result = await container.exec(
                "head -c 8000001 /tmp/result.json", timeout_sec=10
            )
            raw = result.stdout or ""
            if result.return_code or len(raw.encode()) > 8_000_000:
                raise RuntimeError("AI answer worker output limit exceeded")
            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise TypeError("expected object")
                status = data.get("status")
                if status == "ok" and isinstance(data.get("answer"), dict):
                    budget = [30000]
                    return {
                        k: decode(v, budget=budget) for k, v in data["answer"].items()
                    }
                if status not in {"invalid", "unsupported"} or not isinstance(
                    data.get("error"), str
                ):
                    raise ValueError("invalid status")
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise RuntimeError("invalid AI answer worker response") from error
            if status == "unsupported":
                raise UnsupportedAnswerOperation(data["error"])
            raise ValueError(data["error"])
