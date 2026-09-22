import pytest


def test_atomic_json(tmp_path):
    from ddsr_bench.benchmarks.utils import write_json

    path = tmp_path / "result.json"
    path.write_text('{"old": true}')
    with pytest.raises(TypeError):
        write_json(path, {"invalid": object()})
    assert path.read_text() == '{"old": true}'
    assert list(tmp_path.iterdir()) == [path]
