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


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a saved benchmark record."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class Resources:
    """Container limits shared by benchmark task compilers."""

    image: str
    cpus: int
    memory_mb: int
    timeout_sec: float
