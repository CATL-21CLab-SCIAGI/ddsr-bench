from __future__ import annotations

import itertools

import regex
import sympy as sp

_SUPERSCRIPTS = ("dagger", "prime")
_MODIFIER = (
    r"_\{[^}]+\}|_[\\][A-Za-z]+|_[^\^_{}\s]|"
    r"\^\{\\(?:dagger|prime)\}|\^\\(?:dagger|prime)|'+|"
    r"\^\{[^}]+\}|\^[^\^_{}\s]"
)
_NAME = regex.compile(rf"(?P<base>\\?[A-Za-z]+)(?P<mods>(?:{_MODIFIER})+)")
_PART = regex.compile(
    r"('+)|_\{([^}]+)\}|_\\([A-Za-z]+)|_([^\^_{}\s])|"
    r"\^\{\\(dagger|prime)\}|\^\\(dagger|prime)|"
    r"\^\{([^}]+)\}|\^([^\^_{}\s])"
)


def _backslashes(value: str) -> str:
    return regex.sub(r"\\\\+", r"\\", value)


def canonicalize(value: str) -> tuple[str, frozenset[str]]:
    """Flatten LaTeX subscripts and named superscripts into symbol names."""
    introduced = set()

    def rewrite(match: regex.Match) -> str:
        base = match.group("base")
        if base.lstrip("\\") == "int":
            return match.group(0)
        subs = []
        supers = []
        exponents = []
        for part in _PART.finditer(match.group("mods")):
            if part.group(1):
                supers.extend(["prime"] * len(part.group(1)))
            elif part.group(2):
                subs.extend(
                    item.lstrip("\\") for item in regex.split(r"\s*,\s*", part.group(2))
                )
            elif part.group(3) or part.group(4):
                subs.append((part.group(3) or part.group(4)).lstrip("\\"))
            elif part.group(5) or part.group(6):
                token = (part.group(5) or part.group(6)).lstrip("\\")
                supers.append(_NAME.sub(rewrite, token))
            else:
                token = part.group(7) or part.group(8)
                exponents.append(_NAME.sub(rewrite, token))

        name = base
        for token in dict.fromkeys(subs):
            name += f"_{token}"
        unique_supers = []
        for token in supers:
            if token == "prime" or token not in unique_supers:
                unique_supers.append(token)
        for token in unique_supers:
            name += f"_{token}"
        for token in exponents:
            name += f"^{{{token}}}"
        symbol = name.split("^", maxsplit=1)[0]
        if symbol != base:
            introduced.add(symbol)
        return name

    return _backslashes(_NAME.sub(rewrite, value)).strip(), frozenset(introduced)


def _index_rules(value: str) -> tuple[str, dict[str, list[str]]]:
    pairs = regex.findall(r"\(\s*([A-Za-z]\w*)\s*,\s*([^)]+?)\s*\)", value)
    rules = {
        index: [item.lstrip("\\") for item in regex.split(r"\s*,\s*", items)]
        for index, items in pairs
    }
    value = regex.sub(
        r",?\s*\(\s*[A-Za-z]\w*\s*,\s*[^(),]+(?:\s*,\s*[^(),]+)*\s*\)",
        "",
        value,
    )
    return value, rules


def _expand(names: list[str], rules: dict[str, list[str]]) -> list[str]:
    expanded = []
    for name in names:
        present = [
            index
            for index in rules
            if regex.search(rf"_{regex.escape(index)}(?=(?:_|$|\())", name)
        ]
        if not present:
            expanded.append(name)
            continue
        for values in itertools.product(*(rules[index] for index in present)):
            result = name
            for index, replacement in zip(present, values, strict=True):
                result = regex.sub(
                    rf"_{regex.escape(index)}(?=(?:_|$|\())",
                    f"_{replacement}",
                    result,
                )
            expanded.append(result)
    return expanded


def _symbol_names(value: str) -> tuple[list[str], set[str]]:
    value = regex.sub(r"\\lambda", r"\\lam", value)
    noncommuting = regex.findall(r"\(\s*(.+?)\s*,\s*NC\s*\)", value)
    value = regex.sub(r"\(\s*.+?\s*,\s*NC\s*\)", "", value)
    value, rules = _index_rules(value)
    regular = [item.strip() for item in value.replace("$", "").split(";")]
    regular = [item for item in regular if item]
    regular = [canonicalize(_backslashes(item))[0] for item in regular]
    noncommuting = [
        canonicalize(_backslashes(item.strip()))[0] for item in noncommuting
    ]
    all_names = _expand(regular, rules) + _expand(noncommuting, rules)
    return all_names, set(_expand(noncommuting, rules))


def parse_parameters(value: str) -> dict[str, sp.Symbol]:
    """Create the commutative and non-commutative symbols named by a record."""
    names, noncommuting = _symbol_names(value)
    names = [name.replace("\\", "") for name in names]
    noncommuting = {name.replace("\\", "") for name in noncommuting}
    return {
        name: sp.Symbol(name, commutative=name not in noncommuting) for name in names
    }


def parse_functions(value: str) -> dict[str, sp.FunctionClass]:
    """Create the symbolic functions named by a record."""
    names, noncommuting = _symbol_names(value)
    return {
        name: sp.Function(name, commutative=name not in noncommuting) for name in names
    }
