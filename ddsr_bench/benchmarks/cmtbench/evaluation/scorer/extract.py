from __future__ import annotations

import regex

_BOX = regex.compile(r"(boxed|fbox)\{((?:[^{}]|\{(?2)\})*)\}")
_TEXT = regex.compile(r"^\s*\\text\{(.+)\}\s*$", regex.DOTALL)


def extract_solutions(response: str) -> tuple[str, ...]:
    """Extract semicolon-separated expressions from the final answer box."""
    if not response.strip():
        raise ValueError("empty response")
    matches = _BOX.findall(response)
    if not matches or not matches[-1][1].strip():
        raise ValueError("response has no boxed solution")

    content = matches[-1][1]
    wrapper = _TEXT.match(content)
    if wrapper:
        content = wrapper.group(1).strip()
    content = regex.sub(r"\\;", "", content.strip(".,"))
    solutions = tuple(
        part.split(r"\quad")[0].strip(" \n.,") for part in content.split(";")
    )
    if any(not part for part in solutions):
        raise ValueError("boxed solution contains an empty expression")
    return solutions
