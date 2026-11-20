from __future__ import annotations

import logging
import shutil
import tarfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from urllib.parse import quote

import httpx

from .registry import dataset_validator
from .utils import benchmark_config

Fetcher = Callable[[Mapping[str, object], Path], None]
LOGGER = logging.getLogger("ddsr_bench")


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """Canonical recipe for one pinned benchmark edition."""

    name: str
    edition: str
    config: Mapping[str, object]
    fetcher: Fetcher

    def path(self, root: str | Path) -> Path:
        return Path(root) / self.name / self.edition

    def fetch(self, destination: Path) -> None:
        self.fetcher(self.config, destination)

    def validate(self, path: Path) -> None:
        verify_tree(self.config, path)
        dataset_validator(self.name)(path)


@dataclass(frozen=True, slots=True)
class Materialized:
    source: DatasetSource
    path: Path
    reused: bool


@cache
def all_sources() -> dict[str, DatasetSource]:
    """Return every configured, materializable benchmark source."""
    sources = {}
    for name, fetcher in _FETCHERS.items():
        config = benchmark_config(name)
        edition = config.get("edition")
        if not isinstance(edition, str) or not edition:
            raise ValueError(f"benchmark {name!r} requires a dataset edition")
        sources[name] = DatasetSource(name, edition, config, fetcher)
    return sources


def source(name: str) -> DatasetSource:
    try:
        return all_sources()[name]
    except KeyError as error:
        raise ValueError(f"unknown dataset source {name!r}") from error


def fetch_sources(
    names: Iterable[str] | None = None,
    root: str | Path = "datasets",
) -> tuple[Materialized, ...]:
    catalog = all_sources()
    selected = tuple(catalog) if names is None else tuple(names)
    if len(selected) != len(set(selected)):
        raise ValueError("dataset sources must be unique")
    items = tuple(source(name) for name in selected)
    return tuple(_materialize(item, Path(root)) for item in items)


def _materialize(item: DatasetSource, root: Path) -> Materialized:
    destination = item.path(root)
    if destination.exists():
        LOGGER.info("validating existing dataset %s/%s", item.name, item.edition)
        item.validate(destination)
        return Materialized(item, destination, reused=True)

    LOGGER.info("fetching dataset %s/%s", item.name, item.edition)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=f".{item.edition}-", dir=destination.parent))
    try:
        item.fetch(staging)
        item.validate(staging)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            item.validate(destination)
            return Materialized(item, destination, reused=True)
        return Materialized(item, destination, reused=False)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def download(url: str, path: Path) -> None:
    """Stream one source file into an unpublished edition directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(120.0, connect=30.0)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()
        with path.open("xb") as output:
            for chunk in response.iter_bytes():
                output.write(chunk)


def hf_url(repository: str, revision: str, filename: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{quote(repository, safe='/')}/resolve/{quote(revision, safe='')}/"
        f"{quote(filename, safe='/')}?download=true"
    )


def github_url(repository: str, revision: str, filename: str) -> str:
    return (
        f"https://raw.githubusercontent.com/{quote(repository, safe='/')}/"
        f"{quote(revision, safe='')}/{quote(filename, safe='/')}"
    )


def _download_hf(
    config: Mapping[str, object],
    destination: Path,
    files: Iterable[tuple[str, str]],
) -> None:
    repository, revision = str(config["dataset"]), str(config["revision"])
    for remote, local in files:
        download(hf_url(repository, revision, remote), destination / local)


def _download_github(
    config: Mapping[str, object],
    destination: Path,
    files: Iterable[tuple[str, str]],
) -> None:
    repository = str(config.get("repository", config.get("dataset")))
    revision = str(config["revision"])
    for remote, local in files:
        download(github_url(repository, revision, remote), destination / local)


def _github_archive(config: Mapping[str, object], destination: Path) -> None:
    """Materialize one pinned directory from a GitHub source archive."""
    repository = str(config["repository"])
    revision = str(config["repository_revision"])
    selected = Path(str(config["archive_path"]))
    url = f"https://github.com/{repository}/archive/{revision}.tar.gz"
    with NamedTemporaryFile(suffix=".tar.gz") as temporary:
        timeout = httpx.Timeout(120.0, connect=30.0)
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=timeout
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                temporary.write(chunk)
        temporary.flush()
        with tarfile.open(temporary.name, "r:gz") as archive:
            for member in archive.getmembers():
                relative = Path(*Path(member.name).parts[1:])
                if (
                    not member.isfile()
                    or ".." in relative.parts
                    or not relative.is_relative_to(selected)
                ):
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot extract archive member {member.name!r}")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def _hf_file(config: Mapping[str, object], destination: Path) -> None:
    filename = str(config["data_file"])
    _download_hf(config, destination, ((filename, filename),))


def _scicode(config: Mapping[str, object], destination: Path) -> None:
    files = tuple(
        (filename, filename)
        for filename in (str(config["data_file"]), str(config["targets_file"]))
    )
    _download_hf(config, destination, files)


def _hf_files(config: Mapping[str, object], destination: Path) -> None:
    value = config["files"]
    if isinstance(value, Mapping):
        filenames = tuple(
            filename
            for item in value.values()
            for filename in (item if isinstance(item, list) else [item])
        )
    elif isinstance(value, list):
        filenames = tuple(value)
    else:
        raise TypeError("source files must be a list or mapping of names")
    if any(not isinstance(filename, str) for filename in filenames):
        raise TypeError("source file names must be text")
    _download_hf(
        config,
        destination,
        ((filename, filename) for filename in filenames),
    )


def _github_file(config: Mapping[str, object], destination: Path) -> None:
    remote = str(config["data_file"])
    _download_github(config, destination, ((remote, Path(remote).name),))


def _url_file(config: Mapping[str, object], destination: Path) -> None:
    download(str(config["url"]), destination / str(config["data_file"]))


def _critpt(config: Mapping[str, object], destination: Path) -> None:
    directory = str(config["data_dir"])
    count = int(config["expected_challenges"])
    files = (
        (f"{directory}/Challenge_{index}.json", f"Challenge_{index}.json")
        for index in range(1, count + 1)
    )
    _download_github(config, destination, files)


_FETCHERS: dict[str, Fetcher] = {
    "critpt": _critpt,
    "scicode": _scicode,
    "cmphysbench": _hf_file,
    "phybench": _github_file,
    "cmtbench": _hf_file,
    "prismphysics": _hf_files,
    "physics": _hf_files,
    "hle": _hf_files,
    "frontierscience": _hf_files,
    "olympiadbench": _hf_files,
    "hipho": _github_archive,
    "gpqa": _github_file,
    "mmlu": _url_file,
    "mmmupro": _hf_files,
}


def tree_sha256(root: str | Path) -> str:
    """Hash relative file names and file digests in deterministic order."""
    directory = Path(root)
    entries = sorted(directory.rglob("*"))
    if directory.is_symlink() or any(path.is_symlink() for path in entries):
        raise ValueError(f"dataset source {directory} may not contain symlinks")
    files = [path for path in entries if path.is_file()]
    if not files:
        raise ValueError(f"dataset source {directory} has no files")
    digest = sha256()
    for path in files:
        file_digest = sha256()
        with path.open("rb") as source_file:
            while chunk := source_file.read(1024 * 1024):
                file_digest.update(chunk)
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_digest.digest())
    return digest.hexdigest()


def verify_tree(config: Mapping[str, object], root: str | Path) -> None:
    expected = config.get("source_sha256")
    if not isinstance(expected, str) or tree_sha256(root) != expected:
        raise ValueError(f"dataset source {root} does not match its pinned digest")
