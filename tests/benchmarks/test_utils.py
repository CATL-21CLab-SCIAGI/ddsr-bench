import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest

from ddsr_bench.benchmarks.utils import read_bytes, read_json, read_text, write_json


def test_write_json(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"text": "α"})
    original = path.read_bytes()
    for value, error in ((object(), TypeError), (float("nan"), ValueError)):
        with pytest.raises(error):
            write_json(path, {"value": value}, allow_nan=False)
        assert path.read_bytes() == original
        assert list(tmp_path.iterdir()) == [path]
    assert read_json(path) == {"text": "α"}


def test_read_text():
    assert read_text({"text": ""}, "text", "problem") == ""
    with pytest.raises(TypeError, match="problem requires text"):
        read_text({}, "text", "problem")


def test_create_json(tmp_path):
    path = tmp_path / "record.json"

    def create(value):
        try:
            write_json(path, {"value": value}, overwrite=False)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        created = list(pool.map(create, range(4)))
    assert sum(created) == 1
    assert read_json(path) == {"value": created.index(True)}
    assert list(tmp_path.iterdir()) == [path]


def test_failed_create(tmp_path):
    path = tmp_path / "record.json"
    with pytest.raises(ValueError):
        write_json(path, float("nan"), allow_nan=False, overwrite=False)
    assert list(tmp_path.iterdir()) == []


def test_read_json(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"reward": 1}', encoding="utf-8")
    assert read_json(path) == {"reward": 1}
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        read_json(path)
    for content in (b'{"reward":', b"\xff"):
        path.write_bytes(content)
        with pytest.raises(ValueError, match="cannot read"):
            read_json(path)
    with pytest.raises(ValueError, match="cannot read"):
        read_json(tmp_path / "missing.json")


def test_bounded_json(tmp_path):
    path = tmp_path / "record.json"
    raw = '{"text":"α"}'.encode()
    path.write_bytes(raw)
    assert read_json(path, max_bytes=len(raw)) == {"text": "α"}
    with pytest.raises(ValueError, match="input file exceeds"):
        read_json(path, max_bytes=len(raw) - 1)


def test_bounded_read(monkeypatch):
    class Stream(BytesIO):
        def read(self, size=-1):
            assert size == 5  # Read at most the limit plus one, not the entire file.
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda *a, **kw: Stream(b"123456789"))
    with pytest.raises(ValueError, match="input file exceeds"):
        read_bytes(Path("unused"), max_bytes=4)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_json_constants(tmp_path, value):
    path = tmp_path / "record.json"
    path.write_text('{"value":' + value + "}")
    assert isinstance(read_json(path)["value"], float)
    with pytest.raises(ValueError, match=f"nonfinite JSON value: {value}"):
        read_json(path, allow_nan=False)


@pytest.mark.parametrize(
    "raw, error, message",
    [
        (b"{", json.JSONDecodeError, "Expecting property name"),
        (b"\xff", UnicodeDecodeError, "utf-8"),
        (b"[]", TypeError, "expected a JSON object"),
    ],
)
def test_original_errors(tmp_path, raw, error, message):
    path = tmp_path / "record.json"
    path.write_bytes(raw)
    with pytest.raises(error, match=message):
        read_json(path, max_bytes=10, contextual_errors=False)
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "missing", contextual_errors=False)
