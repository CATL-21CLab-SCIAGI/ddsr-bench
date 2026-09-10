from __future__ import annotations

import os


def read_api_key(name: str | None) -> str | None:
    """Read an explicitly selected API key without storing it in configuration."""
    if name is None:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"environment variable {name!r} is not set")
    return value
