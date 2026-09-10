from pathlib import Path
from typing import Any


def trial_fields(result: dict[str, Any], trial: Path) -> tuple[dict[str, Any], Path]:
    """Return CritPt-specific settings and its generated artifact path."""
    agent = (result.get("config") or {}).get("agent") or {}
    style = (agent.get("kwargs") or {}).get("style", "one-step")
    if not isinstance(style, str) or not style:
        raise ValueError("CritPt trial has no generation strategy")
    return {"strategy": style}, trial / "artifacts" / "answer.py"
