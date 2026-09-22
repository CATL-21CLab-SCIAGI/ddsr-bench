import json

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus.batch import _save
from ddsr_bench.benchmarks.critpt.evaluation.consensus.cli import write


@pytest.mark.parametrize("writer", [_save, write])
def test_report_format(tmp_path, writer):
    path = tmp_path / "nested" / "report.json"
    data = {"label": "α", "matched": True}
    writer(path, data)
    expected = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    assert path.read_bytes() == expected.encode("utf-8")
    with pytest.raises(ValueError):
        writer(path.parent / "invalid.json", {"score": float("nan")})
    assert list(path.parent.iterdir()) == [path]


def test_report_overwrite(tmp_path):
    path = tmp_path / "report.json"
    write(path, {"value": 1})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write(path, {"value": 2})
    assert path.read_bytes() == original
    _save(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 2}
    with pytest.raises(ValueError):
        _save(path, {"score": float("inf")})
    assert json.loads(path.read_text()) == {"value": 2}
    assert list(tmp_path.iterdir()) == [path]
