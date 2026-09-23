import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def write_json(
    path: Path,
    value: Any,
    *,
    allow_nan: bool = True,
    overwrite: bool = True,
    newline: bool = False,
) -> None:
    """Publish a complete JSON record; its parent directory must exist.

    Create-only writes use a hard link so competing writers cannot overwrite
    an existing record, even if it appears while serialization is in progress.
    """
    temporary = None
    try:
        with NamedTemporaryFile(
            "w", dir=path.parent, encoding="utf-8", delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=allow_nan)
            if newline:
                file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        if overwrite:
            temporary.replace(path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_text(record: Mapping[str, Any], key: str, label: str) -> str:
    """Read a text field, allowing empty strings where the benchmark permits."""
    value = record.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{label} requires text at {key!r}")
    return value


def read_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Bound the actual read, including files that grow while being read."""
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"input file exceeds {max_bytes / 1_000_000:g} MB")
    return raw


def read_json(
    path: Path,
    *,
    max_bytes: int | None = None,
    allow_nan: bool = True,
    contextual_errors: bool = True,
) -> dict[str, Any]:
    """Read a JSON object, optionally bounding bytes and rejecting NaN/Infinity.

    Defaults preserve ordinary record diagnostics. Set contextual_errors=False
    when an adapter must retain original I/O/decoder errors in per-record reports.
    """

    def reject_constant(value: str):
        raise ValueError(f"nonfinite JSON value: {value}")

    try:
        text = (
            path.read_text(encoding="utf-8")
            if max_bytes is None
            else read_bytes(path, max_bytes=max_bytes).decode("utf-8")
        )
        value = json.loads(text, parse_constant=None if allow_nan else reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if not contextual_errors:
            raise
        raise ValueError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise TypeError(
            f"{path} must contain a JSON object"
            if contextual_errors
            else "expected a JSON object"
        )
    return value


@dataclass(frozen=True, slots=True)
class Resources:
    """Container limits shared by benchmark task compilers."""

    image: str
    cpus: int
    memory_mb: int
    timeout_sec: float
