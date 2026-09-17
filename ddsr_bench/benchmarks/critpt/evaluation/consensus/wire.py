"""Bounded, tagged JSON values. No pickle, eval, or string sympify on outputs."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import sympy as sp
from sympy.core.assumptions import _assume_defined
from sympy.core.function import AppliedUndef

FUNCTIONS = {
    name: getattr(sp, name)
    for name in [
        "Add",
        "Mul",
        "Pow",
        "exp",
        "log",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "Abs",
        "conjugate",
        "re",
        "im",
        "sign",
        "factorial",
        "gamma",
        "KroneckerDelta",
        "Piecewise",
        "Eq",
        "Ne",
        "Lt",
        "Le",
        "Gt",
        "Ge",
        "And",
        "Or",
        "Not",
        "Min",
        "Max",
        "floor",
        "ceiling",
        "binomial",
    ]
}
FUNCTIONS["ExprCondPair"] = sp.functions.elementary.piecewise.ExprCondPair
FUNCTIONS.update(
    {
        "Equality": sp.Eq,
        "Unequality": sp.Ne,
        "StrictLessThan": sp.Lt,
        "LessThan": sp.Le,
        "StrictGreaterThan": sp.Gt,
        "GreaterThan": sp.Ge,
    }
)
CONSTANTS = {
    "pi": sp.pi,
    "E": sp.E,
    "I": sp.I,
    "EulerGamma": sp.EulerGamma,
    "true": sp.true,
    "false": sp.false,
    "True": sp.true,
    "False": sp.false,
}
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def encode(value: Any, depth: int = 0) -> dict:
    if depth > 100:
        raise ValueError("value nesting too deep")
    enc = lambda x: encode(x, depth + 1)
    if isinstance(value, (bool, np.bool_)):
        return {"t": "bool", "v": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"t": "int", "v": str(value)}
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return {"t": "nonfinite"}
        return {"t": "float", "v": repr(float(value))}
    if isinstance(value, (complex, np.complexfloating)):
        return {"t": "complex", "v": [enc(value.real), enc(value.imag)]}
    if isinstance(value, str):
        return {"t": "str", "v": value}
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set, frozenset, sp.FiniteSet)):
        kind = (
            "set" if isinstance(value, (set, frozenset, sp.FiniteSet)) else "sequence"
        )
        items = list(value)
        if kind == "set":
            items.sort(key=str)
        return {"t": kind, "v": [enc(x) for x in items]}
    if isinstance(value, sp.Interval):
        return {
            "t": "interval",
            "v": [enc(value.start), enc(value.end)],
            "open": [bool(value.left_open), bool(value.right_open)],
        }
    if value is sp.S.EmptySet:
        return {"t": "set", "v": []}
    if isinstance(value, sp.Symbol):
        return {
            "t": "symbol",
            "v": value.name,
            "assumptions": {
                k: v for k, v in value.assumptions0.items() if isinstance(v, bool)
            },
        }
    if isinstance(value, sp.Integer):
        return {"t": "int", "v": str(value)}
    if isinstance(value, sp.Rational):
        return {"t": "rational", "v": [str(value.p), str(value.q)]}
    if isinstance(value, sp.Float):
        return {"t": "decimal", "v": str(value)}
    if isinstance(value, sp.Basic):
        if value in (sp.nan, sp.zoo, sp.oo, -sp.oo):
            return {"t": "nonfinite"}
        if str(value) in CONSTANTS and value == CONSTANTS[str(value)]:
            return {"t": "constant", "v": str(value)}
        if isinstance(value, AppliedUndef):
            if value.func.__name__ != "tr":
                raise ValueError("unsupported formal function")
            return {"t": "formal", "v": [enc(x) for x in value.args]}
        name = value.func.__name__
        if name not in FUNCTIONS:
            raise ValueError(f"unsupported symbolic node: {name}")
        return {"t": "expr", "f": name, "v": [enc(x) for x in value.args]}
    if value == sp.Function("tr"):
        return {"t": "function", "v": "tr"}
    raise ValueError(f"unsupported output type: {type(value).__name__}")


def decode(data: dict, depth: int = 0, budget: list[int] | None = None) -> Any:
    budget = budget if budget is not None else [30000]
    budget[0] -= 1
    if depth > 100 or budget[0] < 0 or not isinstance(data, dict):
        raise ValueError("invalid/oversized encoded value")
    dec = lambda x: decode(x, depth + 1, budget)
    kind, value = data.get("t"), data.get("v")
    if kind in {"int", "float", "decimal"}:
        if (
            not isinstance(value, str)
            or len(value) > 2000
            or not NUMBER.fullmatch(value)
        ):
            raise ValueError("invalid encoded number")
        if kind == "int":
            return sp.Integer(value)
        return sp.Float(value, 60)
    if kind == "rational":
        if len(value) != 2 or not all(
            re.fullmatch(r"[+-]?\d{1,2000}", x) for x in value
        ):
            raise ValueError("invalid rational")
        if int(value[1]) == 0:
            raise ValueError("zero rational denominator")
        return sp.Rational(int(value[0]), int(value[1]))
    if kind == "bool" and type(value) is bool:
        return value
    if kind == "str" and isinstance(value, str) and len(value) < 10000:
        return value
    if kind == "nonfinite":
        return sp.nan
    if kind == "complex":
        return dec(value[0]) + sp.I * dec(value[1])
    if kind == "constant":
        return CONSTANTS[value]
    if kind == "symbol":
        if not isinstance(value, str) or len(value) > 100 or "__" in value:
            raise ValueError("invalid symbol")
        assumptions = data.get("assumptions", {})
        if not all(
            k in _assume_defined and type(v) is bool for k, v in assumptions.items()
        ):
            raise ValueError("invalid assumptions")
        return sp.Symbol(value, **assumptions)
    if kind in {"sequence", "set", "expr", "formal", "interval"}:
        if not isinstance(value, list) or len(value) > 10000:
            raise ValueError("invalid encoded children")
        args = [dec(x) for x in value]
        if kind == "sequence":
            return tuple(args)
        if kind == "set":
            return frozenset(args)
        if kind == "interval":
            flags = data.get("open")
            if (
                len(args) != 2
                or not isinstance(flags, list)
                or len(flags) != 2
                or not all(type(x) is bool for x in flags)
            ):
                raise ValueError("invalid interval")
            return sp.Interval(*args, left_open=flags[0], right_open=flags[1])
        if kind == "formal":
            return sp.Function("tr")(*args)
        return FUNCTIONS[data["f"]](*args)
    if kind == "function" and value == "tr":
        return sp.Function("tr")
    raise ValueError("unsupported encoded value")
