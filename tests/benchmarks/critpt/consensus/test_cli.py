import json

import pytest

from ddsr_bench.benchmarks.critpt.evaluation.consensus import cli


@pytest.fixture
def replay(monkeypatch, sample_bundle_path):
    # Exercise command output without executing reference or model code.
    monkeypatch.setattr(cli, "DiagnosticRuntime", lambda **kwargs: None)
    monkeypatch.setattr(cli, "provenance", lambda *args: {"label": "α"})
    monkeypatch.setattr(cli.Grader, "grade", lambda *args: {"status": "matched"})

    def run(path):
        return cli.main(
            [
                "replay",
                "--references",
                str(sample_bundle_path),
                "--output",
                str(path),
                "--problems",
                "1",
            ]
        )

    return run


def test_report_format(tmp_path, replay):
    path = tmp_path / "nested" / "report.json"
    assert replay(path) == 0
    data = json.loads(path.read_text())
    assert data["summary"]["healthy"] is True
    expected = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    assert path.read_bytes() == expected.encode("utf-8")


def test_report_nonfinite(tmp_path, replay, monkeypatch):
    monkeypatch.setattr(cli, "provenance", lambda *args: {"value": float("nan")})
    path = tmp_path / "report.json"
    assert replay(path) == 2
    assert not path.exists()


def test_report_overwrite(tmp_path, replay):
    path = tmp_path / "report.json"
    assert replay(path) == 0
    original = path.read_bytes()
    files = set(tmp_path.iterdir())
    assert replay(path) == 2
    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == files
