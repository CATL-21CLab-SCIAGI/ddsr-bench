"""Behavior-compatible implementation of the official CMT scorer."""

from .extract import extract_solutions
from .functional import equivalent_functional
from .latex import normalize_latex
from .numeric import equivalent, parse_numeric
from .symbolic import evaluate_parameterized, parse_expression, parse_symbolic
from .symbols import canonicalize, parse_functions, parse_parameters

__all__ = [
    "canonicalize",
    "equivalent",
    "equivalent_functional",
    "evaluate_parameterized",
    "extract_solutions",
    "normalize_latex",
    "parse_expression",
    "parse_functions",
    "parse_numeric",
    "parse_parameters",
    "parse_symbolic",
]
