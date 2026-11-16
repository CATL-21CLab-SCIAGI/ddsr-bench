import json
from pathlib import Path

from ddsr_bench.benchmarks.prismphysics.data.loader import THRESHOLD, load_problem
from ddsr_bench.benchmarks.prismphysics.evaluation.verifier import verify

FIXTURE = Path(__file__).parents[2] / "fixtures" / "prismphysics_02_3006.json"


def answer():
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return load_problem(record, "02").answer


def test_full_credit() -> None:
    reference = answer()
    assert reference is not None

    result = verify(r"$$C=\frac{\varepsilon_0L^2}{d}$$", reference, THRESHOLD)

    assert result["reward"] == 1
    assert result["status"] == "passed"
    assert result["correct"] is True
    assert result["direct_matches"] == [2]
    assert r"\varepsilon_0" not in json.dumps(result)


def test_partial_credit() -> None:
    reference = answer()
    assert reference is not None

    result = verify(r"$$C=\frac{\varepsilon S}{d}$$", reference, THRESHOLD)

    assert result["reward"] == 0.5
    assert result["status"] == "different"
    assert result["correct"] is False


def test_timeout() -> None:
    reference = answer()
    assert reference is not None

    result = verify("anything", reference, THRESHOLD, timeout_sec=0)

    assert result["reward"] == 0
    assert result["status"] == "timeout"
    assert result["error"] == "TimeoutError"
