import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schemas import AIProblemSpec, CritPtSource, ProblemSpec


def encode_instruction(problem: ProblemSpec | AIProblemSpec) -> str:
    """Serialize public fields as JSON for Harbor's required instruction.md."""
    data = asdict(problem)
    data.pop("source_path")
    return json.dumps(data, indent=2, ensure_ascii=False)


def decode_instruction(instruction: str) -> ProblemSpec:
    """Restore the public problem fields from one prepared task."""
    data = json.loads(instruction)
    data["source"] = CritPtSource(data.get("source"))
    spec = AIProblemSpec if data["source"] == CritPtSource.AI else ProblemSpec
    return spec(**data, source_path=Path("prepared-task"))


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
