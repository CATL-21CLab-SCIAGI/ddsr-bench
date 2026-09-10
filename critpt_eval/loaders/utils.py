import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and report its path when parsing fails."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def require_text(record: dict, key: str, path: Path) -> str:
    """Read one required, non-empty string from a record."""
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} requires non-empty text at {key!r}")
    return value
