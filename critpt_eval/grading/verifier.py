from __future__ import annotations

import copy
import inspect
import json
import runpy
from pathlib import Path
from typing import Any

from .compare import compare
from .validation import validate_code


def _functions(
    answer_path: Path, reference_path: Path
) -> tuple[Any, Any, dict[str, Any]]:
    answer = runpy.run_path(str(answer_path)).get("answer")
    reference_namespace = runpy.run_path(str(reference_path))
    reference = reference_namespace.get("real_answer")
    if not callable(answer) or not callable(reference):
        raise TypeError("answer and real_answer must be callable")
    answer_signature = inspect.signature(answer)
    reference_signature = inspect.signature(reference)
    answer_parameters = tuple(
        (name, parameter.kind)
        for name, parameter in answer_signature.parameters.items()
    )
    reference_parameters = tuple(
        (name, parameter.kind)
        for name, parameter in reference_signature.parameters.items()
    )
    if answer_parameters != reference_parameters:
        raise TypeError("answer signature does not match real_answer")
    return answer, reference, reference_namespace


def _call(
    function: Any,
    signature: inspect.Signature,
    namespace: dict[str, Any],
    *,
    exact: bool = False,
) -> Any:
    args = []
    kwargs = {}
    parameters = tuple(signature.parameters.values())
    if exact and set(namespace) != {parameter.name for parameter in parameters}:
        raise ValueError("testcase keys do not match the answer signature")
    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError("variable arguments are not supported")
        if parameter.name not in namespace:
            if parameter.default is inspect.Parameter.empty:
                raise ValueError(f"no verifier value for parameter {parameter.name!r}")
            continue
        argument_value = namespace[parameter.name]
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(argument_value)
        else:
            kwargs[parameter.name] = argument_value
    return function(*args, **kwargs)


def _testcases(path: Path) -> list[dict[str, Any]]:
    try:
        testcases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot load testcases") from error
    if (
        not isinstance(testcases, list)
        or not testcases
        or any(not isinstance(testcase, dict) for testcase in testcases)
    ):
        raise ValueError("testcases must be a non-empty list of objects")
    return testcases


def verify(
    answer_path: Path,
    template_path: Path,
    reference_path: Path | None = None,
    testcases_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one answer and compare it with an optional local reference."""
    mode = "reference" if reference_path is not None else "format"
    try:
        code = answer_path.read_text(encoding="utf-8")
        template = template_path.read_text(encoding="utf-8")
        validate_code(code, template)
        if reference_path is None:
            if testcases_path is not None:
                raise ValueError("testcases require a reference answer")
            return {"reward": 1.0, "mode": mode, "status": "passed"}

        answer, reference, namespace = _functions(answer_path, reference_path)
        signature = inspect.signature(reference)
        testcases = _testcases(testcases_path) if testcases_path else [namespace]
        results = []
        for testcase in testcases:
            exact = testcases_path is not None
            reference_inputs = copy.deepcopy(testcase) if exact else testcase
            answer_inputs = copy.deepcopy(testcase) if exact else testcase
            expected = _call(reference, signature, reference_inputs, exact=exact)
            actual = _call(answer, signature, answer_inputs, exact=exact)
            results.append(compare(actual, expected))
        passed = sum(results)
        return {
            "reward": float(passed == len(results)),
            "mode": mode,
            "status": "passed" if passed == len(results) else "different",
            "passed": passed,
            "total": len(results),
        }
    except Exception as error:  # noqa: BLE001 - model failures become zero reward
        return {
            "reward": 0.0,
            "mode": mode,
            "status": "error",
            "error": type(error).__name__,
        }


def main() -> None:
    tests = Path("/tests")
    reference = tests / "reference.py"
    testcases = tests / "testcases.json"
    result = verify(
        Path("/app/answer.py"),
        tests / "template.py",
        reference if reference.exists() else None,
        testcases if testcases.exists() else None,
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (logs / "reward.txt").write_text(str(result["reward"]), encoding="utf-8")


if __name__ == "__main__":
    main()
