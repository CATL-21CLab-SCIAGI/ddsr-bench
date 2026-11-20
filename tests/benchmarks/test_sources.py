from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from ddsr_bench.benchmarks import sources


class Adapter:
    fetched = 0

    @classmethod
    def fetch(cls, config: object, destination: Path) -> None:
        cls.fetched += 1
        (destination / "data.txt").write_text("valid", encoding="utf-8")

    @staticmethod
    def validate(path: Path) -> None:
        if (path / "data.txt").read_text(encoding="utf-8") != "valid":
            raise ValueError("invalid dataset")


def _digest(content: bytes) -> str:
    return sha256(b"data.txt\0" + sha256(content).digest()).hexdigest()


def _catalog(content: bytes = b"valid") -> dict[str, sources.DatasetSource]:
    config = {"source_sha256": _digest(content)}
    return {"demo": sources.DatasetSource("demo", "v1", config, Adapter.fetch)}


def _mock_catalog(monkeypatch: pytest.MonkeyPatch, content: bytes = b"valid") -> None:
    monkeypatch.setattr(sources, "all_sources", lambda: _catalog(content))
    monkeypatch.setattr(sources, "dataset_validator", lambda name: Adapter.validate)


def test_fetches_then_reuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    Adapter.fetched = 0
    _mock_catalog(monkeypatch)

    first = sources.fetch_sources(("demo",), tmp_path)[0]
    second = sources.fetch_sources(("demo",), tmp_path)[0]

    assert first.path == tmp_path / "demo" / "v1"
    assert not first.reused and second.reused
    assert Adapter.fetched == 1


def test_does_not_replace_invalid_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "demo" / "v1"
    destination.mkdir(parents=True)
    (destination / "data.txt").write_text("invalid", encoding="utf-8")
    Adapter.fetched = 0
    _mock_catalog(monkeypatch, b"invalid")

    with pytest.raises(ValueError, match="invalid dataset"):
        sources.fetch_sources(("demo",), tmp_path)

    assert Adapter.fetched == 0
    assert (destination / "data.txt").read_text(encoding="utf-8") == "invalid"


def test_removes_failed_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(config: object, path: Path) -> None:
        raise RuntimeError("failed")

    catalog = {
        "demo": sources.DatasetSource(
            "demo",
            "v1",
            {"source_sha256": "unused"},
            fail,
        )
    }
    monkeypatch.setattr(sources, "all_sources", lambda: catalog)

    with pytest.raises(RuntimeError, match="failed"):
        sources.fetch_sources(("demo",), tmp_path)

    assert list((tmp_path / "demo").iterdir()) == []


def test_rejects_unknown_source_before_fetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Adapter.fetched = 0
    _mock_catalog(monkeypatch)

    with pytest.raises(ValueError, match="unknown dataset source"):
        sources.fetch_sources(("demo", "missing"), tmp_path)

    assert Adapter.fetched == 0


def test_tree_digest_includes_paths_and_bytes(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"bc")
    nested = tmp_path / "ab"
    nested.mkdir()
    (nested / "c").write_bytes(b"")

    assert sources.tree_sha256(tmp_path) == (
        "5feac4b7d17a5cc7c693cae2a95dcfad0e03d2139683156598a913a7f3b3ad33"
    )


def test_tree_rejects_symlinks(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("data", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to("data.txt")

    with pytest.raises(ValueError, match="symlinks"):
        sources.tree_sha256(tmp_path)

    root = tmp_path.parent / "linked-dataset"
    root.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        sources.tree_sha256(root)


def test_catalog_has_pinned_digests() -> None:
    catalog = sources.all_sources()

    assert set(catalog) == {
        "critpt",
        "scicode",
        "cmphysbench",
        "phybench",
        "cmtbench",
        "prismphysics",
        "physics",
        "hle",
        "frontierscience",
        "olympiadbench",
        "hipho",
        "gpqa",
    }
    assert all(
        len(str(item.config.get("source_sha256"))) == 64 for item in catalog.values()
    )
