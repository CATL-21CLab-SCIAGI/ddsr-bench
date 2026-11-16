from __future__ import annotations

import json
from dataclasses import asdict
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.cmtbench.data.schemas import AnswerSpec
from ddsr_bench.benchmarks.utils import write_json

from .evaluator import evaluate


def _worker(connection: Connection, response: str, answer: AnswerSpec) -> None:
    try:
        result = evaluate(response, answer)
        connection.send(("ok", asdict(result) | {"reward": result.reward}))
    except Exception as error:  # noqa: BLE001 - worker failures receive zero
        connection.send(("error", type(error).__name__))
    finally:
        connection.close()


def _failure(
    error: str,
    status: str = "error",
    method: str | None = None,
) -> dict[str, Any]:
    result = {
        "reward": 0.0,
        "mode": "cmtbench",
        "status": status,
        "error": error,
    }
    if method is not None:
        result["method"] = method
    return result


def verify(
    response: str,
    answer: AnswerSpec,
    timeout_sec: float = 120,
) -> dict[str, Any]:
    """Score one response in a bounded worker without executing model code."""
    if timeout_sec <= 0:
        return _failure("TimeoutError", "timeout")

    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, response, answer))
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
    if not evaluation["success"]:
        return _failure(evaluation["error"], method=evaluation["method"])
    passed = evaluation["equivalent"] is True
    return {
        "reward": evaluation["reward"],
        "mode": "cmtbench",
        "status": "passed" if passed else "different",
        "method": evaluation["method"],
        "success": True,
        "equivalent": passed,
    }


def main() -> None:
    tests = Path("/tests")
    reference = json.loads((tests / "reference.json").read_text(encoding="utf-8"))
    answer = AnswerSpec(
        solution=reference["solution"],
        parameters=reference["parameters"],
        functions=reference["functions"],
    )
    result = verify(
        Path("/app/answer.txt").read_text(encoding="utf-8"),
        answer,
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    write_json(logs / "result.json", result)
    (logs / "reward.txt").write_text(str(result["reward"]), encoding="utf-8")


if __name__ == "__main__":
    main()
