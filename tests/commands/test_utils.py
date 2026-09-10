import pytest

from ddsr_bench.commands.utils import read_api_key


def test_reads_selected_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret")

    assert read_api_key("TEST_API_KEY") == "secret"
    assert read_api_key(None) is None


def test_requires_selected_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MISSING_API_KEY"):
        read_api_key("MISSING_API_KEY")
