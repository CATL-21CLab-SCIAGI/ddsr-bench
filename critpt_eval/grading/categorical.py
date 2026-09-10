from typing import Any


def categorical(actual: Any, expected: Any) -> bool:
    """Compare two non-empty multiple-choice labels."""
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    left = actual.strip().casefold()
    right = expected.strip().casefold()
    return bool(left) and left == right
