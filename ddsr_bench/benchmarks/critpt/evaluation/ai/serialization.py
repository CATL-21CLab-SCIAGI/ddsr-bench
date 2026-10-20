"""AI worker values use the grader's functions, not consensus's narrower set."""

from functools import partial

import sympy as sp
from sympy.functions.special.hyper import TupleArg

from .. import serialization
from .validation import _ALLOWED_SYMPY_CALLS

# Only symbolic constructors from the trusted grader allowlist. Never resolve
# arbitrary names from worker output; utilities such as simplify are not nodes.
# Canonical names also cover aliases (for example digamma -> polygamma).
_FUNCTIONS = {
    **serialization.FUNCTIONS,
    **{
        function.__name__: function
        for name in _ALLOWED_SYMPY_CALLS
        if isinstance(function := getattr(sp, name, None), sp.FunctionClass)
    },
    "Tuple": sp.Tuple,  # hypergeometric functions contain symbolic tuples.
    "TupleArg": TupleArg,
}
encode = partial(serialization.encode, preserve_literals=True, functions=_FUNCTIONS)
decode = partial(serialization.decode, functions=_FUNCTIONS)
