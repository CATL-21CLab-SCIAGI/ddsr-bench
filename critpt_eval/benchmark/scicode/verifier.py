from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

_SKIPPED = {"13.6", "62.1", "76.3"}


def _cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("SciCode cases must be a list")
    for case in data:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or not isinstance(case.get("tests"), list)
            or any(not isinstance(test, str) for test in case["tests"])
        ):
            raise TypeError("each SciCode step requires an ID and text tests")
    return data


def _script(solution: str, case: dict[str, Any], h5_path: Path) -> str:
    tests = case["tests"]
    package_root = Path(__file__).resolve().parents[3]
    lines = [
        solution,
        "import sys",
        f"sys.path.insert(0, {str(package_root)!r})",
        "from critpt_eval.benchmark.scicode.loader import load_targets",
        f"targets = load_targets({case['id']!r}, {len(tests)}, {str(h5_path)!r})",
    ]
    for index, test in enumerate(tests):
        lines.extend((f"target = targets[{index}]", test))
    return "\n\n".join(lines) + "\n"


def verify(
    solution_path: Path,
    cases_path: Path,
    h5_path: Path,
    *,
    timeout: float = 1_800,
) -> dict[str, Any]:
    """Run official SciCode assertions, isolated once more per step."""
    try:
        solution = solution_path.read_text(encoding="utf-8")
        cases = [case for case in _cases(cases_path) if case["id"] not in _SKIPPED]
        if not h5_path.is_file():
            raise FileNotFoundError(f"SciCode targets not found at {h5_path}")

        steps = []
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for case in cases:
                script = directory / f"{case['id']}.py"
                script.write_text(_script(solution, case, h5_path), encoding="utf-8")
                try:
                    process = subprocess.run(
                        [sys.executable, "-I", str(script)],
                        capture_output=True,
                        check=False,
                        timeout=timeout,
                    )
                    status = "passed" if process.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    status = "timeout"
                steps.append({"id": case["id"], "status": status})

        passed = sum(step["status"] == "passed" for step in steps)
        complete = passed == len(steps)
        return {
            "reward": float(complete),
            "mode": "scicode",
            "status": "passed" if complete else "different",
            "passed": passed,
            "total": len(steps),
            "steps": steps,
        }
    except Exception as error:  # noqa: BLE001 - candidate failures earn zero
        return {
            "reward": 0.0,
            "mode": "scicode",
            "status": "error",
            "error": type(error).__name__,
        }


def main() -> None:
    result = verify(
        Path("/app/solution.py"),
        Path("/tests/cases.json"),
        Path("/opt/scicode/test_data.h5"),
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (logs / "reward.txt").write_text(str(result["reward"]), encoding="utf-8")


if __name__ == "__main__":
    main()
