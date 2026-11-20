from __future__ import annotations

from multiprocessing import get_context
from multiprocessing.connection import Connection
from typing import Any

from ddsr_bench.benchmarks.olympiadbench.data.schemas import AnswerSpec, ProblemSpec

from .scorer import equivalent


def evaluate(response: str, problem: ProblemSpec, answer: AnswerSpec) -> dict[str, Any]:
    """Apply official automatic scoring where the benchmark defines it."""
    metadata = {
        "kind": problem.kind,
        "modality": problem.modality,
        "subject": problem.subject,
        "language": problem.language,
        "source": problem.source,
    }
    if problem.kind == "TP":
        return {
            "reward": None,
            "mode": "olympiadbench",
            "status": "ungraded",
            "gradable": False,
            **metadata,
        }
    if not response.strip() or not answer.answers:
        return {
            "reward": 0.0,
            "mode": "olympiadbench",
            "status": "error",
            "error": "NoResponse" if not response.strip() else "MissingReference",
            "gradable": True,
            **metadata,
        }
    precision: float | list[float]
    if answer.tolerance is None:
        precision = 1e-8
    else:
        values = [
            float(value) if value.strip() else 1e-8
            for value in answer.tolerance.split(",")
        ]
        precision = values[0] if len(values) == 1 else values
    reference = ",".join(answer.answers)
    passed = equivalent(reference, response, precision)
    return {
        "reward": float(passed),
        "mode": "olympiadbench",
        "status": "passed" if passed else "different",
        "gradable": True,
        **metadata,
    }


def _worker(
    connection: Connection,
    response: str,
    problem: ProblemSpec,
    answer: AnswerSpec,
) -> None:
    try:
        connection.send(evaluate(response, problem, answer))
    except Exception as error:  # noqa: BLE001 - scorer failures are trial errors
        connection.send({"error": type(error).__name__})
    finally:
        connection.close()


def verify(
    response: str,
    problem: ProblemSpec,
    answer: AnswerSpec,
    timeout_sec: float = 120,
) -> dict[str, Any]:
    """Run symbolic scoring in a bounded worker process."""
    if problem.kind == "TP":
        return evaluate(response, problem, answer)
    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, response, problem, answer))
    process.start()
    sender.close()
    process.join(timeout_sec)
    if process.is_alive():
        process.terminate()
        process.join()
        result = evaluate("", problem, answer)
        result["error"] = "TimeoutError"
    elif receiver.poll():
        result = receiver.recv()
        if set(result) == {"error"}:
            failure = evaluate("", problem, answer)
            failure["error"] = result["error"]
            result = failure
    else:
        result = evaluate("", problem, answer)
        result["error"] = "WorkerError"
    receiver.close()
    process.close()
    return result
