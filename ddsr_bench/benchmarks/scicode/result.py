from pathlib import Path
from typing import Any


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    """Return SciCode-specific settings and its generated artifact path."""
    agent = (result.get("config") or {}).get("agent") or {}
    with_background = (agent.get("kwargs") or {}).get("with_background", False)
    if not isinstance(with_background, bool):
        raise TypeError("SciCode with_background must be boolean")
    return {"with_background": with_background}, trial / "artifacts" / "solution.py"
