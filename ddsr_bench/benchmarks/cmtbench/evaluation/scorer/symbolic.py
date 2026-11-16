from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import regex
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .extract import extract_solutions
from .latex import normalize_latex
from .numeric import Number
from .symbols import canonicalize, parse_functions, parse_parameters

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass(frozen=True, slots=True)
class Parsed:
    expressions: tuple[sp.Expr, ...]
    normalized: tuple[str, ...]
    parameters: dict[str, sp.Symbol]
    functions: dict[str, sp.FunctionClass]


def parse_expression(value: str, symbols: dict[str, object]) -> sp.Expr:
    """Parse one normalized expression using upstream's transformations."""
    if "=" in value:
        value = value.split("=")[-1].strip()
    if not value:
        raise ValueError("empty symbolic expression")
    value = regex.sub(r"\s{2,}", " ", value)
    try:
        return parse_expr(value, local_dict=symbols, transformations=_TRANSFORMATIONS)
    except Exception as error:
        raise ValueError("cannot parse symbolic expression") from error


def parse_symbolic(
    response: str,
    parameter_text: str = "",
    function_text: str = "",
) -> Parsed:
    """Extract and parse every expression in one boxed response."""
    parameters = parse_parameters(parameter_text)
    functions = parse_functions(function_text)
    symbols: dict[str, object] = {**parameters, **functions}
    normalized = []
    expressions = []
    for solution in extract_solutions(response):
        canonical, introduced = canonicalize(solution)
        for name in introduced:
            clean = name.replace("\\", "")
            # Upstream overwrites declared symbols introduced by sub/superscript
            # flattening, even when the declaration was non-commutative.
            symbols[clean] = sp.Symbol(clean)
        expression = normalize_latex(canonical, symbols)
        normalized.append(expression)
        expressions.append(parse_expression(expression, symbols))
    return Parsed(
        expressions=tuple(expressions),
        normalized=tuple(normalized),
        parameters=parameters,
        functions=functions,
    )


def evaluate_parameterized(response: str, parameter_text: str) -> tuple[Number, ...]:
    """Evaluate symbolic answers at upstream's deterministic parameter values."""
    parsed = parse_symbolic(response, parameter_text)
    random = np.random.RandomState(42)
    values = {symbol: random.uniform(1, 2) for symbol in parsed.parameters.values()}
    evaluated = []
    for expression in parsed.expressions:
        value = expression.subs(values).evalf()
        try:
            evaluated.append(float(value))
        except TypeError:
            evaluated.append(complex(value))
    return tuple(evaluated)
