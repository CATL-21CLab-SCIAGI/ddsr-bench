from __future__ import annotations

import json
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.prismphysics.data.schemas import AnswerSpec, FormulaNode
from ddsr_bench.benchmarks.utils import write_json


def _worker(
    connection: Connection,
    response: str,
    answer: AnswerSpec,
    threshold: float,
) -> None:
    try:
        # Import and contain the upstream-derived engine inside this worker: it
        # changes NumPy/warning globals and may print diagnostics while parsing.
        from .evaluator import evaluate

        with redirect_stdout(StringIO()):
            result = evaluate(response, answer, threshold)
        connection.send(("ok", asdict(result)))
    except Exception as error:  # noqa: BLE001 - worker failures receive zero
        connection.send(("error", type(error).__name__))
    finally:
        connection.close()


def _failure(error: str, status: str = "error") -> dict[str, Any]:
    return {
        "reward": 0.0,
        "mode": "prismphysics",
        "status": status,
        "error": error,
    }


def verify(
    response: str,
    answer: AnswerSpec,
    threshold: float,
    timeout_sec: float = 120,
) -> dict[str, Any]:
    """Score one response in a bounded worker without executing model code."""
    if timeout_sec <= 0:
        return _failure("TimeoutError", "timeout")

    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker, args=(sender, response, answer, threshold)
    )
    process.start()
    sender.close()
    process.join(timeout_sec)
    if process.is_alive():
        process.terminate()
        process.join()
        receiver.close()
        process.close()
        return _failure("TimeoutError", "timeout")

    payload = receiver.recv() if receiver.poll() else ("error", "WorkerError")
    receiver.close()
    process.close()
    if payload[0] == "error":
        return _failure(payload[1])

    evaluation = payload[1]
    for field in ("direct_matches", "credited", "matches"):
        evaluation[field] = list(evaluation[field])
    return {
        "reward": evaluation["score"],
        "mode": "prismphysics",
        "status": "passed" if evaluation["correct"] else "different",
        "success": True,
        "threshold": threshold,
        **evaluation,
    }


def main() -> None:
    tests = Path("/tests")
    reference = json.loads((tests / "reference.json").read_text(encoding="utf-8"))
    answer = AnswerSpec(
        tuple(
            FormulaNode(
                index=node["index"],
                formula=node["formula"],
                dependencies=tuple(node["dependencies"]),
                final=node["final"],
            )
            for node in reference["formulas"]
        )
    )
    result = verify(
        Path("/app/answer.txt").read_text(encoding="utf-8"),
        answer,
        threshold=float(reference["threshold"]),
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    write_json(logs / "result.json", result)
    (logs / "reward.txt").write_text(str(result["reward"]), encoding="utf-8")


if __name__ == "__main__":
    main()
