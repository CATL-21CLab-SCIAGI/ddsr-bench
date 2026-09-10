from __future__ import annotations

import ast
import re
from collections import Counter

_UNSAFE_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Yield,
    ast.YieldFrom,
)
_UNSAFE_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
_UNSAFE_MODULES = {
    "asyncio",
    "httpx",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
_UNSAFE_ATTRIBUTES = {
    "fromfile",
    "load",
    "load_library",
    "loadtxt",
    "memmap",
    "open",
    "popen",
    "read",
    "read_text",
    "request",
    "save",
    "savetxt",
    "system",
    "tofile",
    "write",
    "write_text",
    "urlopen",
}


def extract_code(response: str) -> str:
    """Extract code using CritPt's first-Python-block convention."""
    if "```python" in response:
        code = response.split("```python", 1)[1].split("```", 1)[0]
    elif re.search(r"```\s*\n", response):
        code = response.split("```", 1)[1].split("```", 1)[0]
    elif "```" not in response:
        code = response
    else:
        code = ""
    if not code.strip():
        raise ValueError("response does not contain Python code")
    return code.strip() + "\n"


def _parse(code: str, label: str) -> ast.Module:
    try:
        return ast.parse(code)
    except SyntaxError as error:
        raise ValueError(f"{label} has invalid Python syntax: {error.msg}") from error


def _imports(tree: ast.Module) -> Counter[tuple[str, str, str | None, int]]:
    found: Counter[tuple[str, str, str | None, int]] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update((alias.name, "", alias.asname, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.update(
                (node.module or "", alias.name, alias.asname, node.level)
                for alias in node.names
            )
    return found


def validate_code(code: str, template: str) -> None:
    """Validate generated code against its trusted CritPt template."""
    tree = _parse(code, "answer")
    template_tree = _parse(template, "code template")
    allowed_imports = _imports(template_tree)
    imports = _imports(tree)
    if imports - allowed_imports:
        raise ValueError("answer may only use imports provided by the code template")

    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    if (
        len(functions) != 1
        or functions[0].name != "answer"
        or functions[0] not in tree.body
    ):
        raise ValueError("answer must define exactly one answer() function")
    template_functions = [
        node
        for node in template_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "answer"
    ]
    if len(template_functions) != 1:
        raise ValueError("code template must define exactly one answer() function")
    if ast.dump(functions[0].args) != ast.dump(template_functions[0].args):
        raise ValueError("answer signature does not match the code template")

    for module, _, _, _ in imports:
        if module.split(".")[0] in _UNSAFE_MODULES:
            raise ValueError(f"answer may not import {module!r}")
    for node in ast.walk(tree):
        if isinstance(node, _UNSAFE_NODES):
            raise TypeError(f"answer uses unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and (
            node.id in _UNSAFE_NAMES or node.id.startswith("__")
        ):
            raise ValueError(f"answer may not use {node.id!r}")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr in _UNSAFE_ATTRIBUTES
        ):
            raise ValueError(f"answer may not access {node.attr!r}")


def extract_answer(response: str, template: str) -> str:
    code = extract_code(response)
    validate_code(code, template)
    return code
