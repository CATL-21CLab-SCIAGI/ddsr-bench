"""Use the same code extraction and template contract as generation."""

from __future__ import annotations

import ast

from ddsr_bench.grading.validation import extract_code, validate_code

source = extract_code
validate = validate_code


def parameters(template: str) -> list[str]:
    fn = next(
        n
        for n in ast.parse(template).body
        if isinstance(n, ast.FunctionDef) and n.name == "answer"
    )
    return [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
