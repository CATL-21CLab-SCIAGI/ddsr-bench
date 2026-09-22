from __future__ import annotations

import json
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from ddsr_bench.benchmarks.phybench.evaluation.eed import score
from ddsr_bench.benchmarks.utils import write_json


def _escaped(text: str, index: int) -> bool:
    slashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        slashes += 1
        index -= 1
    return slashes % 2 == 1


def extract_boxed(text: str) -> str:
    """Extract the final balanced box described by PHYBench's paper."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return ""

    start += len(marker)
    depth = 1
    for end in range(start, len(text)):
        if _escaped(text, end):
            continue
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start:end].strip()
    return ""


def normalize_reference(text: str) -> str:
    """Remove the display wrappers retained in PHYBench's public data."""
    value = text.strip()
    if r"\boxed{" in value:
        return extract_boxed(value)
    for opening, closing in (("$$", "$$"), (r"\[", r"\]"), ("$", "$")):
        if value.startswith(opening) and value.endswith(closing):
            return value[len(opening) : -len(closing)].strip()
    return value


def _worker(connection: Connection, reference: str, prediction: str) -> None:
    try:
        with redirect_stdout(StringIO()):
            result = score(reference, prediction)
        connection.send(("ok", asdict(result)))
    except Exception as error:  # noqa: BLE001 - scorer failures receive zero
        connection.send(("error", type(error).__name__))
    finally:
        connection.close()


def _failure(error: str, status: str = "error") -> dict[str, Any]:
    return {"reward": 0.0, "mode": "eed", "status": status, "error": error}


def verify(
    response: str,
    reference: str,
    timeout_sec: float = 120,
) -> dict[str, Any]:
    """Extract and score one response in an isolated local worker."""
    prediction = extract_boxed(response)
    reference = normalize_reference(reference)
    if not prediction:
        return _failure("MissingBox")
    if not reference:
        return _failure("MissingReference")
    if timeout_sec <= 0:
        return _failure("TimeoutError", "timeout")

    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, reference, prediction))
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
    accurate = result["points"] == 100
    return {
        "reward": result["points"] / 100,
        "mode": "eed",
        "status": "passed" if accurate else "different",
        "eed_score": result["points"],
        "accurate": accurate,
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
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    write_json(logs / "result.json", result)
    (logs / "reward.txt").write_text(str(result["reward"]), encoding="utf-8")


if __name__ == "__main__":
    main()
