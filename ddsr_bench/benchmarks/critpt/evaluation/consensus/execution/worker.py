"""Isolated JSON worker. Evaluation receives one code file, never other answers."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import random
import resource
import sys

import numpy as np

from ddsr_bench.grading.validation import validate_code

from .serialization import decode, encode


def nonfinite_value(value: dict) -> bool:
    if value["t"] == "nonfinite":
        return True
    if value["t"] in {"sequence", "set", "complex", "interval"}:
        return any(nonfinite_value(child) for child in value["v"])
    return False


def evaluate(payload: dict) -> dict:
    try:
        validate_code(payload["code"], payload["template"])
    except (ValueError, TypeError, SyntaxError, StopIteration) as error:
        return {"status": "error", "stage": "validation", "error": str(error)[:300]}
    try:
        namespace = {"__name__": "consensus_answer"}
        random.seed(0)
        np.random.seed(0)
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            exec(  # noqa: S102 - isolated execution
                compile(payload["code"], "answer.py", "exec"), namespace
            )
            results = []
            for case in payload["inputs"]:
                kwargs = {}
                for key, value in case.items():
                    if value["t"] == "float":
                        kwargs[key] = float(value["v"])
                    elif value["t"] == "int":
                        kwargs[key] = int(value["v"])
                    else:
                        kwargs[key] = decode(value)
                value = encode(namespace["answer"](**kwargs))
                if nonfinite_value(value):
                    raise ValueError("nonfinite returned value")
                results.append(value)
        return {"status": "ok", "outputs": results}
    except Exception as error:  # noqa: BLE001 - model error boundary
        return {
            "status": "error",
            "stage": "execution",
            "error": f"{type(error).__name__}: {str(error)[:300]}",
        }


def compare(payload: dict) -> dict:
    from ..matching.compare import Comparator, combine

    answer = [decode(x) for x in payload["answer"]]
    results = []
    for reference in payload["references"]:
        try:
            expected = [decode(x) for x in reference["outputs"]]
            if len(answer) != len(expected):
                raise ValueError("case counts differ")
            checks = [
                Comparator(payload["number"], payload["parameters"], i).compare(a, b)
                for i, (a, b) in enumerate(zip(answer, expected))
            ]
            result = combine(checks).json()
        except Exception as error:  # noqa: BLE001 - comparison boundary
            result = {
                "status": "unknown",
                "methods": [],
                "reason": f"comparison error: {type(error).__name__}",
            }
        results.append({"reference": reference["id"], **result})
        if result["status"] == "matched":
            break
    return {"status": "ok", "comparisons": results}


def main() -> None:
    limit = int(os.environ.get("CONSENSUS_CPU_SECONDS", "90"))
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 1))
    raw = sys.stdin.buffer.read(8_000_001)
    if len(raw) > 8_000_000:
        raise ValueError("worker request too large")
    payload = json.loads(raw)
    try:
        result = (
            evaluate(payload) if payload["action"] == "evaluate" else compare(payload)
        )
    except Exception as error:  # noqa: BLE001 - bounded worker protocol
        result = {"status": "error", "stage": "worker", "error": type(error).__name__}
    result["runtime"] = {
        "python": sys.version.split()[0],
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("sympy", "numpy", "scipy")
        },
    }
    encoded = json.dumps(result, allow_nan=False)
    if len(encoded) > 8_000_000:
        encoded = json.dumps(
            {"status": "error", "stage": "output", "error": "output too large"}
        )
    print(encoded)


if __name__ == "__main__":
    main()
