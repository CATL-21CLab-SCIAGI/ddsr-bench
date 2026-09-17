"""Six composable comparison modes, with explicit evidence and unknown outcomes."""

from __future__ import annotations

import signal
from contextlib import contextmanager
from dataclasses import dataclass

import mpmath as mp
import sympy as sp

from .fixtures import continuation, sample_values, symbols
from .policy import SIGNIFICANT_DIGITS, Precision, precision


class Undecided(Exception):
    pass


@contextmanager
def deadline(seconds: float = 2.0):
    def expired(*_):
        raise Undecided("symbolic operation timed out")

    previous = signal.signal(signal.SIGALRM, expired)
    timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
        signal.signal(signal.SIGALRM, previous)


@dataclass(frozen=True)
class Check:
    status: str
    methods: tuple[str, ...] = ()
    reason: str = ""
    details: dict | None = None

    def json(self) -> dict:
        result = {
            "status": self.status,
            "methods": list(self.methods),
            "reason": self.reason,
        }
        if self.details is not None:
            result["details"] = self.details
        return result


def combine(checks: list[Check], method: str = "") -> Check:
    methods = tuple(
        sorted({m for c in checks for m in c.methods} | ({method} if method else set()))
    )
    for status in ("different", "unknown"):
        if any(c.status == status for c in checks):
            failed = next(c for c in checks if c.status == status)
            return Check(status, methods, failed.reason, failed.details)
    return Check("matched", methods)


def finite(value) -> bool:
    return isinstance(value, sp.Basic) and not value.has(sp.nan, sp.zoo, sp.oo, -sp.oo)


def numeric(a, b, rule: Precision, significant: int | None = None) -> Check:
    try:
        a, b = sp.sympify(a), sp.sympify(
            b
        )  # already-decoded objects, never output strings
        if not finite(b):
            return Check("unknown", reason="nonfinite reference")
        if not finite(a):
            return Check("different", reason="nonfinite candidate")
        if rule.exact:
            with deadline():
                difference = a - b
                okay = difference == 0 or difference.is_zero is True
                return Check(
                    "matched" if okay else "different",
                    ("exact",),
                    "" if okay else "exact values differ",
                )
        with deadline(), mp.workdps(60):

            def number(x):
                z = sp.N(x, 60)
                if z.free_symbols:
                    raise Undecided("numeric value contains symbols")
                return mp.mpc(str(sp.re(z)), str(sp.im(z)))

            left, right = number(a), number(b)
            if not (mp.isfinite(left) and mp.isfinite(right)):
                raise Undecided("nonfinite numeric evaluation")
            delta = abs(left - right)
            if rule.exact:
                okay = delta == 0
                method = "exact"
            else:
                tolerance = max(
                    mp.mpf(str(rule.absolute)),
                    mp.mpf(str(rule.relative)) * max(abs(left), abs(right)),
                )
                if significant is not None and right:
                    tolerance = mp.mpf("0.5") * mp.power(
                        10, mp.floor(mp.log10(abs(right))) - significant + 1
                    )
                okay = delta <= tolerance + mp.mpf("1e-45") * max(abs(left), abs(right))
                method = "numeric_tolerance"
            return Check(
                "matched" if okay else "different",
                (method,),
                "" if okay else "numeric values differ",
            )
    except (TypeError, ValueError, Undecided):
        return Check("unknown", reason="numeric evaluation undecidable")


class Comparator:
    def __init__(self, number: int, parameter_names: list[str], case: int = 0):
        self.number = number
        self.case = case
        self.canonical = {
            x.name: x
            for x in symbols(number, parameter_names).values()
            if isinstance(x, sp.Symbol)
        }

    def expression(self, expr):
        if not isinstance(expr, sp.Basic):
            expr = sp.sympify(expr)
        replacements = {}
        for sym in expr.free_symbols:
            if sym.name not in self.canonical:
                raise Undecided(f"unexpected free symbol: {sym.name}")
            replacements[sym] = self.canonical[sym.name]
        expr = expr.xreplace(replacements)
        if self.number == 62:
            k = self.canonical["k"]
            if self.case == 0:
                expr = expr.subs(k, 1)
        return expr

    def _normalize(self, expr, path):
        n = self.number
        with deadline(3):
            if n == 2 and path == (0,):
                expr = sp.series(expr, self.canonical["sigma2"], 0, 2).removeO()
            elif n == 59:
                expr = sp.series(expr, self.canonical["epsilon"], 0, 2).removeO()
            elif n == 66:
                q = self.canonical["q"]
                if expr.has(sp.log(q)) or expr.subs(q, 0).has(
                    sp.zoo, sp.oo, -sp.oo, sp.nan
                ):
                    raise Undecided("unsupported nonanalytic generating function")
                expr = sp.series(expr, q, 0, 16).removeO()
                if not expr.is_polynomial(q):
                    raise Undecided("non-polynomial truncated generating function")
            elif n == 23:
                for key in ("epsilon_UV", "epsilon_IR"):
                    expr = sp.series(
                        expr,
                        self.canonical[key],
                        0,
                        1,
                        dir="-" if key == "epsilon_IR" else "+",
                    ).removeO()
            elif n == 49:
                z = self.canonical["z"]
                expr = sp.series(expr, z, sp.oo, 1).removeO()
                expr = sp.expand(sp.expand_log(expr, force=False))
                expr = sp.Add(*(term for term in sp.Add.make_args(expr) if term.has(z)))
        return expr

    def _point(self, expr, values: dict):
        subs = {s: values.get(s.name, values["_default"]) for s in expr.free_symbols}
        with deadline():
            result = expr.subs(subs, simultaneous=True)
            if result.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                for name, bounds in continuation(self.number).items():
                    sym = self.canonical.get(name)
                    if sym is None or sym not in expr.free_symbols:
                        continue
                    for point, direction in bounds:
                        if subs.get(sym) == point:
                            rest = {s: v for s, v in subs.items() if s != sym}
                            result = sp.limit(
                                expr.subs(rest, simultaneous=True),
                                sym,
                                point,
                                dir=direction,
                            )
                            if finite(result):
                                return sp.N(result, 50)
                raise Undecided("undefined value at a configured point")
            return sp.N(result, 50)

    def _sample(self, a, b, path, boundary_only=False):
        checks = []
        for values in sample_values(self.number, self.case, path):
            if boundary_only and not any(
                values.get(name) == point
                for name, bounds in continuation(self.number).items()
                for point, _ in bounds
            ):
                continue
            try:
                left, right = self._point(a, values), self._point(b, values)
                check = numeric(left, right, Precision(1e-10, 1e-35))
            except (Undecided, ValueError, TypeError, NotImplementedError):
                check = Check(
                    "unknown", reason="configured point could not be evaluated"
                )
            checks.append(check)
            if check.status == "different":
                return Check(
                    "different",
                    ("sampled",),
                    "counterexample at a fixed domain point",
                    {
                        "path": list(path),
                        "point": {
                            k: str(v) for k, v in values.items() if k != "_default"
                        },
                        "actual": str(left),
                        "expected": str(right),
                    },
                )
        if not checks:
            return (
                Check("matched", ("boundary",))
                if boundary_only
                else Check("unknown", reason="no domain samples")
            )
        result = combine(checks, "boundary" if boundary_only else "sampled")
        return result

    def _symbolic(self, a, b, path):
        try:
            a, b = self.expression(a), self.expression(b)
            if not finite(b):
                return Check("unknown", reason="nonfinite reference expression")
            if not finite(a):
                return Check("different", reason="nonfinite candidate expression")
            a, b = self._normalize(a, path), self._normalize(b, path)
            # Check explicitly supplied endpoint values before interior simplification.
            boundary = (
                self._sample(a, b, path, True)
                if continuation(self.number)
                else Check("matched")
            )
            if boundary.status != "matched":
                return boundary
            if self.number == 49:
                z = self.canonical["z"]
                ell = sp.Dummy("log_z")
                with deadline():
                    pa = sp.Poly(sp.expand(a).subs(sp.log(z), ell), z, ell)
                    pb = sp.Poly(sp.expand(b).subs(sp.log(z), ell), z, ell)
                aa, bb = pa.as_dict(), pb.as_dict()
                return combine(
                    [
                        self._plain_symbolic(
                            aa.get(k, sp.S.Zero), bb.get(k, sp.S.Zero), path
                        )
                        for k in set(aa) | set(bb)
                    ],
                    "asymptotic_coefficients",
                )
            if self.number == 66:
                q = self.canonical["q"]
                return combine(
                    [
                        numeric(
                            sp.expand(a).coeff(q, i),
                            sp.expand(b).coeff(q, i),
                            Precision(exact=True),
                        )
                        for i in range(16)
                    ],
                    "series_coefficients",
                )
            result = self._plain_symbolic(a, b, path)
            method = "series" if self.number in {2, 23, 59} else ""
            return combine([boundary, result], method)
        except (
            Undecided,
            ValueError,
            TypeError,
            NotImplementedError,
            sp.PolynomialError,
        ):
            return Check(
                "unknown", reason="unsupported or timed-out symbolic normalization"
            )

    def _plain_symbolic(self, a, b, path):
        if a == b:
            return Check("matched", ("symbolic",))
        if not (a.free_symbols | b.free_symbols):
            return numeric(a, b, Precision(1e-12, 0.0))
        if self.number == 65:
            # tr is formal; scalar probes would erase operator distinctions.
            return Check("different", ("formal",), "different canonical operators")
        for operation in (
            lambda x: sp.cancel(sp.together(x)),
            sp.trigsimp,
            sp.simplify,
        ):
            try:
                with deadline():
                    difference = operation(a - b)
                    if difference == 0:
                        return Check("matched", ("symbolic",))
                    if (
                        difference.is_number
                        and difference.is_zero is False
                        and not difference.has(sp.Float)
                    ):
                        return Check(
                            "different", ("symbolic",), "nonzero constant difference"
                        )
            except (Undecided, ValueError, TypeError, NotImplementedError):
                pass
        if self.number == 39:
            alpha = self.canonical["alpha"]
            u, v = sp.symbols("_real_alpha _imag_alpha", real=True)
            try:
                with deadline():
                    if sp.simplify((a - b).subs(alpha, u + sp.I * v)) == 0:
                        return Check("matched", ("symbolic_complex",))
            except Undecided:
                pass
        return self._sample(a, b, path)

    def _sets(self, a, b, path):
        if len(a) != len(b):
            return Check("different", reason="set cardinalities differ")
        left, right = sorted(a, key=str), sorted(b, key=str)
        if self.number == 40 and len(a) == 2:
            return combine(
                [
                    self._symbolic(sum(left), sum(right), path),
                    self._symbolic(sp.prod(left), sp.prod(right), path),
                ],
                "root_polynomial",
            )
        edges = [
            [self.compare(x, y, path + (j,)) for j, y in enumerate(right)] for x in left
        ]

        def matching(allow_unknown):
            owners = {}

            def visit(i, visited):
                for j, edge in enumerate(edges[i]):
                    if j in visited or edge.status not in (
                        {"matched", "unknown"} if allow_unknown else {"matched"}
                    ):
                        continue
                    visited.add(j)
                    if j not in owners or visit(owners[j], visited):
                        owners[j] = i
                        return True
                return False

            if all(visit(i, set()) for i in range(len(left))):
                return [edges[i][j] for j, i in owners.items()]
            return None

        found = matching(False)
        if found is not None:
            return combine(found, "set_bijection")
        if matching(True) is not None:
            return Check("unknown", reason="set pairing is undecided")
        return Check("different", reason="no bijective set match")

    def compare(self, a, b, path=()) -> Check:
        if isinstance(b, bool):
            return Check(
                "matched" if type(a) is bool and a == b else "different", ("exact",)
            )
        if isinstance(b, str):
            return Check(
                (
                    "matched"
                    if isinstance(a, str)
                    and a.strip().casefold() == b.strip().casefold()
                    else "different"
                ),
                ("exact",),
            )
        if isinstance(b, tuple):
            if not isinstance(a, tuple) or len(a) != len(b):
                return Check("different", reason="sequence shape differs")
            return combine(
                [self.compare(x, y, path + (i,)) for i, (x, y) in enumerate(zip(a, b))],
                "ordered",
            )
        if isinstance(b, frozenset):
            if not isinstance(a, frozenset):
                return Check("different", reason="expected a set")
            return self._sets(a, b, path)
        if isinstance(b, sp.Interval):
            if (
                not isinstance(a, sp.Interval)
                or a.left_open != b.left_open
                or a.right_open != b.right_open
            ):
                return Check("different", reason="interval type or openness differs")
            return combine(
                [
                    self._symbolic(a.start, b.start, path + (0,)),
                    self._symbolic(a.end, b.end, path + (1,)),
                ],
                "interval",
            )
        if isinstance(a, (bool, str, tuple, frozenset, sp.Interval)):
            return Check("different", reason="wrong scalar type")
        if not isinstance(a, sp.Expr) or not isinstance(b, sp.Expr):
            return Check("unknown", reason="unsupported scalar type")
        # Parameterized outputs use tight expression comparison even when a leaf
        # happens to be numeric. Fixed-value tasks use field-specific precision.
        if self.canonical and self.number not in {20, 45}:
            return self._symbolic(a, b, path)
        rule = precision(self.number, path)
        if self.number == 35 and path == (40,):
            rule = Precision(0, 1e-10)
        if self.number == 43 and not (-sp.Rational(1, 2) <= a < sp.Rational(1, 2)):
            return Check("different", reason="charge outside canonical interval")
        if self.number == 63 and not (0 <= a <= 1):
            return Check("different", reason="efficiency outside [0,1]")
        return numeric(a, b, rule, SIGNIFICANT_DIGITS.get(self.number))
