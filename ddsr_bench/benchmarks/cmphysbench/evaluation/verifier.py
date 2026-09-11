from __future__ import annotations

import json
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.cmphysbench.data.schemas import AnswerType
from ddsr_bench.benchmarks.cmphysbench.evaluation.seed import score


def extract_boxed(text: str) -> str:
    """Extract the first balanced boxed answer exactly as CMPhysBench does."""
    start = text.find(r"\boxed{")
    if start < 0:
        return ""
    start += len(r"\boxed{")
    depth = 1
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start:end].strip()
    return ""


def _worker(
    connection: Connection,
    reference: str,
    prediction: str,
    answer_type: AnswerType,
) -> None:
    try:
        with redirect_stdout(StringIO()):
            result = score(reference, prediction, answer_type)
        data = asdict(result) | {
            "reward": result.reward,
            "accurate": result.accurate,
        }
        connection.send(("ok", data))
    except Exception as error:  # noqa: BLE001 - scorer failures receive zero
        connection.send(("error", type(error).__name__))
    finally:
        connection.close()


def _failure(error: str, status: str = "error") -> dict[str, Any]:
    return {
        "reward": 0.0,
        "mode": "seed",
        "status": status,
        "error": error,
    }


def verify(
    response: str,
    reference: str,
    answer_type: AnswerType,
    timeout_sec: float = 120,
) -> dict[str, Any]:
    """Extract and score one response in an isolated local worker."""
    prediction = extract_boxed(response)
    if not prediction:
        return _failure("MissingBox")
    if timeout_sec <= 0:
        return _failure("TimeoutError", "timeout")

    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(sender, reference, prediction, answer_type),
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

    result = payload[1]
    return {
        "reward": result["points"] / 100,
        "mode": "seed",
        "status": "passed" if result["accurate"] else "different",
        "seed_score": result["points"],
        "accurate": result["accurate"],
        "relative_distance": result["relative_distance"],
        "tree_size": result["tree_size"],
        "raw_distance": result["raw_distance"],
        "prediction": prediction,
    }


def main() -> None:
    reference = json.loads(Path("/tests/reference.json").read_text(encoding="utf-8"))
    result = verify(
        Path("/app/answer.txt").read_text(encoding="utf-8"),
        reference["answer"],
        reference["answer_type"],
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (logs / "reward.txt").write_text(str(result["reward"]), encoding="utf-8")


if __name__ == "__main__":
    main()
