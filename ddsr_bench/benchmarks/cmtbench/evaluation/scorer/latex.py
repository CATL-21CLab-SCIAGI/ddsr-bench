from __future__ import annotations

from collections.abc import Iterable

import regex

_FUNCTIONS = tuple(
    sorted(
        {
            "Ai",
            "Ei",
            "abs",
            "acos",
            "acosh",
            "acot",
            "acoth",
            "acsc",
            "acsch",
            "airyai",
            "asec",
            "asech",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "cos",
            "cosh",
            "cot",
            "coth",
            "csc",
            "csch",
            "erf",
            "erfc",
            "erfi",
            "exp",
            "gamma",
            "ln",
            "log",
            "max",
            "min",
            "sec",
            "sech",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
        },
        key=len,
        reverse=True,
    )
)
_DELETIONS = (
    r"\\left",
    r"\\right",
    r"\\Bigl",
    r"\\Bigr",
    r"\\bigl",
    r"\\bigr",
    r"\\biggl",
    r"\\biggr",
    r"\\Biggl",
    r"\\Biggr",
    r"\\Big",
    r"\\big",
    r"\\Bigg",
    r"\\bigg",
    r"\\hline",
    r"\\vline",
    r"\$",
    r"\\pm",
    r"\\langle",
    r"\\rangle",
    r"\\,",
    r"\\;",
    r"~",
    r"\n",
    r"\\!",
    r"\\displaystyle",
    r"\\:",
    r"\\quad",
    r"\\qquad",
    r"\\begin\{aligned\}",
    r"\\end\{aligned\}",
)
_NESTED = (
    (r"\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)"),
    (r"\\(?:t)?frac(\d)(\d)", r"(\1)/(\2)"),
    (r"\\sqrt\[(.*?)\]\{([^{}]*)\}", r"(\2)**(1/(\1))"),
    (r"\\sqrt\{([^{}]*)\}", r"(\1)**(1/2)"),
    (r"\^\{([^{}]*)\}", r"^(\1)"),
    (r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1"),
    (r"\|([^{}]*)\|", r"abs(\1)"),
)


def _parenthesize_functions(value: str) -> str:
    names = "|".join(map(regex.escape, _FUNCTIONS))
    value = regex.sub(rf"(?<![A-Za-z])?\\?({names})(?![A-Za-z])", r" \1", value)
    pattern = regex.compile(
        rf"((?:\\)?(?:{names}))(?![A-Za-z_])\s*"
        r"(?:\{([^}]+?)\}|\[([^]]+?)\]|([^{}\[\]\s()+\-*/^]+))"
    )

    def replace(match: regex.Match) -> str:
        argument = match.group(2) or match.group(3) or match.group(4)
        return f"{match.group(1)}({argument})"

    return pattern.sub(replace, value)


def _lex_identifiers(value: str, known_names: Iterable[str]) -> str:
    """Lexically tokenize names and separate adjacent identifiers upstream-style.

    Here ``lex`` means lexical analysis: scanning an expression into identifier
    tokens. It is distinct from the preceding LaTeX-to-expression rewrites.
    """
    names = sorted(set(known_names) | set(_FUNCTIONS) | {"pi"}, key=len, reverse=True)
    token = regex.compile("(" + "|".join(map(regex.escape, names)) + r"|[A-Za-z])")
    parts = []
    previous_identifier = False
    position = 0
    while position < len(value):
        match = token.match(value, position)
        if match is None:
            parts.append(value[position])
            previous_identifier = False
            position += 1
            continue
        identifier = match.group(1)
        if identifier in ("e", "i"):
            identifier = identifier.upper()
        if previous_identifier:
            parts.append("*")
        parts.append(identifier)
        previous_identifier = True
        position = match.end()
    return "".join(parts)


def normalize_latex(value: str, known_names: Iterable[str] = ()) -> str:
    """Apply the pinned scorer's core LaTeX rewrites."""
    if not value.strip():
        raise ValueError("empty LaTeX expression")
    # Upstream computes Unicode substitutions on a discarded copy. Preserve the
    # observed behavior here; changing it would alter official parser results.
    current = value
    for pattern in _DELETIONS:
        current = regex.sub(pattern, "", current)
    # Upstream's sub/sup preprocessing trims deletion residue before later
    # rewrites, while preserving spaces introduced by function rewrites below.
    current = current.strip()

    names = "|".join(map(regex.escape, _FUNCTIONS))
    for wrapper in ("text", "mathrm", "operatorname", "displaystyle"):
        current = regex.sub(rf"\\{wrapper}\{{\s*({names})\s*\}}", r"\1", current)
    current = regex.sub(r"Ai\(([^)]+)\)", r"airyai(\1)", current)
    for pattern, replacement in (
        (r"exp", r"E^"),
        (r"\\(?:cdot|times)", r"*"),
        (r"\\pi", r"pi"),
        (r"\\i", r" I"),
        (r"\\Gamma\((.*?)\)", r"gamma(\1)"),
    ):
        current = regex.sub(pattern, replacement, current)

    for _ in range(5):
        previous = current
        for pattern, replacement in _NESTED:
            current = regex.sub(pattern, replacement, current)
        if current == previous:
            break
    current = _parenthesize_functions(current)
    for pattern, replacement in (
        (r"\\(?:approx|sim)", "="),
        (r"\\\{", "("),
        (r"\\\}", ")"),
        (r"\[", "("),
        (r"\]", ")"),
        (r"\{-\}", "-"),
        (r"\{\+\}", "+"),
        (r"\\lambda", r"\\lam"),
    ):
        current = regex.sub(pattern, replacement, current)
    current = regex.sub(r"(?<![a-zA-Z])e(?![a-zA-Z])", "E", current)
    current = regex.sub(r"([a-zA-Z])e\^", r"\1*E^", current)
    current = regex.sub(r"[{}]", lambda match: "(" if match[0] == "{" else ")", current)
    current = _lex_identifiers(current, known_names)
    current = regex.sub(r"\\", " ", current)
    current = regex.sub(r"\s+", " ", current)
    if not current.strip():
        raise ValueError("LaTeX conversion produced an empty expression")
    return current
