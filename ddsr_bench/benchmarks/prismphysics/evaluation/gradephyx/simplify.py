import re

DEBUGGING = False
FUNCTION_FOR_ADDING_TIMES_BEFORE_NON_FUNCTIONAL_BRACKETS = (
    True  # must be true, just use a if so you can fold it.
)
if FUNCTION_FOR_ADDING_TIMES_BEFORE_NON_FUNCTIONAL_BRACKETS:
    import regex  # pip install regex

    EXCLUDED = {"frac", "sqrt", "times", "cdot"}

    # ------------------------------------------------------------
    # 1  Pattern – no more “++” inside the char class
    # ------------------------------------------------------------
    PATTERN = regex.compile(
        r"""
        (\\?[a-zA-Z]\w*)\s*        # 1 – variable or \command
        \(                         #    opening (
            (                      # 2 – balanced contents
                (?:
                    [^()]          #    any non‑paren char  ← now back‑trackable
                | (?R)           #    or another balanced (…)  (recursion)
                )*
            )
        \)                         #    closing )
        """,
        regex.VERBOSE,
    )

    # ------------------------------------------------------------
    # 2  Helpers
    # ------------------------------------------------------------
    def is_valid_var(v: str) -> bool:
        return v.lstrip("\\") not in EXCLUDED

    def repl(m: regex.Match) -> str:
        var, inner = m.group(1), m.group(2)

        # If the "variable" is actually an excluded command (e.g. \frac),
        # just keep searching inside its parentheses.
        if not is_valid_var(var):
            return f"{var}({regex.sub(PATTERN, repl, inner)})"

        # Only insert × if arithmetic symbols are present inside ( … )
        if regex.search(r"[\+\-\*/]|\\frac|\\cdot|\\times", inner):
            return f"{var} \\times ({inner})"

        return m.group(0)  # leave untouched

    def add_times_before_NonFunctionalbrackets(expr: str) -> str:
        """Keep applying the substitution until nothing changes."""
        prev = None
        counter = 0
        while expr != prev:
            counter += 1
            prev = expr
            expr = regex.sub(PATTERN, repl, expr)
            if counter > 100:
                break
        return expr

else:

    def add_times_before_NonFunctionalbrackets(expr: str) -> str:
        return


# -------------------------------------------------------------------
# Debug logging
# -------------------------------------------------------------------
def _log_debug(filepath: str, expr: str):
    """Append expr to filepath if DEBUGGING is True."""
    if not DEBUGGING:
        return
    with open(filepath, "a") as f:
        f.write(expr + "\n")


# -------------------------------------------------------------------
# 1. Remove \left and \right wrappers
# -------------------------------------------------------------------
def _remove_left_right(expr: str) -> str:
    # Strip \left and \right (any amount of whitespace after)
    expr = re.sub(r"\\left\s*", "", expr)
    expr = re.sub(r"\\right\s*", "", expr)
    return expr


# -------------------------------------------------------------------
# 2. Collapse line breaks and hard spaces
# -------------------------------------------------------------------
def _remove_line_breaks(expr: str) -> str:
    # (\\\\) is a LaTeX linebreak, \newline or explicit " \ "
    return re.sub(r"(\\\\|\\newline|\\\s)", " ", expr)


# -------------------------------------------------------------------
# 3. Remove spacing commands and tildes
# -------------------------------------------------------------------
def _remove_spacings_and_tildes(expr: str) -> str:
    # Removes \, \; \: \!  and multiple spaces, then strips '~'
    expr = re.sub(r"\\[ ,;:!]", "", expr)
    expr = expr.replace("~", "")
    # collapse any remaining multi‑spaces
    return re.sub(r"\s+", " ", expr).strip()


# -------------------------------------------------------------------
# 4. Strip common math environments
# -------------------------------------------------------------------
def _remove_environments(expr: str) -> str:
    # Remove \begin{...} and \end{...} for simple env names
    expr = re.sub(r"\\begin\{[a-zA-Z*]+\}", "", expr)
    expr = re.sub(r"\\end\{[a-zA-Z*]+\}", "", expr)
    return expr


# -------------------------------------------------------------------
# 5. Flatten text macros (\mathrm, \operatorname, etc.)
# -------------------------------------------------------------------
def _flatten_text_macros(expr: str) -> str:
    # \mathrm{foo} → foo
    expr = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", expr)
    expr = re.sub(r"\\operatorname\{([^}]*)\}", r"\1", expr)
    expr = re.sub(r"\\text\{([^}]*)\}", r"\1", expr)
    return expr


# -------------------------------------------------------------------
# 6. Strip trailing punctuation
# -------------------------------------------------------------------
def _strip_trailing_punct(expr: str) -> str:
    # If the end is not a digit or ')', drop trailing comma or period
    expr = re.sub(r"([^\d)])[,\.]\s*$", r"\1", expr)
    return expr.strip()


# -------------------------------------------------------------------
# 7. Replace common LaTeX constructs
# -------------------------------------------------------------------
def _replace_common_patterns(expr: str) -> str:
    # Change \approx to =
    expr = re.sub(r"\\approx", "=", expr)
    # \ddot{X} → \ddot_X
    expr = re.sub(r"\\ddot\{([^\{\}]+)\}", r"\\ddot_\1", expr)
    return expr


_SCRIPT_OPERATORS = {
    "+": r"\notion_plus",
    "-": r"\notion_minus",
    "*": r"\notion_times",
    "/": r"\notion_div",
    "=": r"\notion_equals",
}


def _matching_open(value: str, closing: int) -> int | None:
    depth = 0
    for index in range(closing, -1, -1):
        if value[index] == "}":
            depth += 1
        elif value[index] == "{":
            depth -= 1
            if depth == 0:
                return index
    return None


def _unwrap_script_atoms(value: str) -> str:
    """Match PRISM's removal of simple braces immediately before scripts."""
    index = 0
    while index < len(value):
        if value[index] != "}":
            index += 1
            continue
        script = index + 1
        while script < len(value) and value[script].isspace():
            script += 1
        if script == len(value) or value[script] not in "^_":
            index += 1
            continue
        opening = _matching_open(value, index)
        if opening is None:
            index += 1
            continue
        atom = value[opening + 1 : index].strip()
        plain = len(atom) == 1
        command = atom.startswith("\\") and atom[1:].isalpha()
        invalid = any(
            character.isspace() or character in "{}+-*/=" for character in atom
        )
        if not atom or invalid or not (plain or command):
            index += 1
            continue
        prefix = value[:opening]
        command_start = opening - 1
        while command_start >= 0 and value[command_start].isalpha():
            command_start -= 1
        if (
            command_start >= 0
            and value[command_start] == "\\"
            and not prefix.endswith(" ")
        ):
            prefix += " "
        value = prefix + atom + value[index + 1 :]
        index = len(prefix) + len(atom)
    return value


def _protect_script_operators(value: str) -> str:
    """Encode a bare operator used as an entire subscript or superscript."""
    output = []
    index = 0
    while index < len(value):
        marker = value[index]
        if marker not in "^_":
            output.append(marker)
            index += 1
            continue
        output.append(marker)
        index += 1
        while index < len(value) and value[index].isspace():
            index += 1
        if index < len(value) and value[index] == "{":
            depth = 0
            closing = index
            while closing < len(value):
                depth += value[closing] == "{"
                depth -= value[closing] == "}"
                if depth == 0:
                    break
                closing += 1
            if closing < len(value):
                content = value[index + 1 : closing].strip()
                replacement = _SCRIPT_OPERATORS.get(content)
                output.append(
                    "{" + replacement + "}"
                    if replacement
                    else value[index : closing + 1]
                )
                index = closing + 1
                continue
        replacement = (
            _SCRIPT_OPERATORS.get(value[index]) if index < len(value) else None
        )
        output.append(replacement or (value[index] if index < len(value) else ""))
        index += index < len(value)
    return "".join(output)


# -------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------
def simplify_latex_expr(expr: str) -> str:
    # input logging
    _log_debug("org_expr.txt", expr)

    expr = _remove_left_right(expr)
    expr = _remove_line_breaks(expr)
    expr = _remove_spacings_and_tildes(expr)
    expr = _remove_environments(expr)
    expr = _flatten_text_macros(expr)
    expr = _strip_trailing_punct(expr)
    expr = _replace_common_patterns(expr)
    expr = _unwrap_script_atoms(expr)
    expr = _protect_script_operators(expr)

    # finally, insert explicit times “×”
    expr = add_times_before_NonFunctionalbrackets(expr)

    # output logging
    _log_debug("ex_expr.txt", expr)
    return expr
