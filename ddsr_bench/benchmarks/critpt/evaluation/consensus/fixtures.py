"""Deterministic input cases and domain samples for the reviewed 63 problems."""

from __future__ import annotations

import sympy as sp

from .validation import parameters

ALIASES = {"np": "n_prime", "lambda_": "lambda"}
POSITIVE = {
    2: "k_plus k_minus alpha vbar_b",
    3: "m r0",
    7: "n d",
    11: "K",
    15: "N l",
    18: "epsilon0 k z_R d0 m Omega_1 Omega_2",
    23: "p_z epsilon_UV mu",
    24: "mu",
    29: "lambda E W alpha m hbar",
    36: "k n X_tot",
    38: "T",
    39: "g gamma",
    49: "z K",
    51: "g lambda",
    59: "M a",
    60: "Delta_k_sq",
    62: "k",
    67: "d",
}
INTEGERS = {
    7: "n d",
    11: "m",
    15: "N l",
    36: "n X_tot",
    39: "n n_prime",
    51: "g",
    59: "M",
    62: "k",
    67: "d",
}


def symbols(number: int, names: list[str]) -> dict:
    result = {}
    for argument in names:
        name = ALIASES.get(argument, argument)
        if name == "tr":
            result[argument] = sp.Function("tr")
            continue
        if name == "k_value":
            continue
        assumptions = {}
        if name in POSITIVE.get(number, "").split():
            assumptions["positive"] = True
        elif number == 39 and name in {"n", "n_prime"}:
            assumptions["nonnegative"] = True
        elif number in {23, 24} and name == "epsilon_IR":
            assumptions["negative"] = True
        elif not ((number == 39 and name == "alpha") or number in {65, 66}):
            assumptions["real"] = True
        if name in INTEGERS.get(number, "").split():
            assumptions["integer"] = True
        result[argument] = sp.Symbol(name, **assumptions)
    return result


def inputs(number: int, template: str) -> list[dict]:
    if number == 20:
        result = []
        for i, (a, b, wavelength, distance) in enumerate(
            [
                (80, 40, 1064, 2000),
                (90, 30, 780, 2500),
                (60, 20, 1064, 3500),
                (70, 45, 1550, 3000),
                (100, 50, 780, 4000),
                (75, 25, 1064, 2200),
                (85, 35, 1550, 5000),
                (65, 30, 780, 1800),
            ]
        ):
            result.append(
                {
                    "a": float(a),
                    "b": float(b),
                    "rho": float(1800 + 200 * i),
                    "k": float(2 * sp.pi / (wavelength * 1e-9)),
                    "epsilon_r": 2.1 + 0.1 * i,
                    "P_0": 10.0 + 5 * i,
                    "w_0": 700.0 + 50 * i,
                    "R": float(distance),
                }
            )
        return result
    if number == 45:
        return [
            dict(zip(("mc_x", "mc_y", "mv_x", "mv_y"), map(float, row)))
            for row in [
                (1, 2, 2, 4),
                (1, 1, 1, 1),
                (2, 4, 1, 2),
                (1, 2, 1, 3),
                (1, 3, 1, 2),
                (4, 2, 3, 5),
                (0.5, 2, 1, 4),
                (0.5, 2, 1, 3),
                (1, 1, 1, 1.000001),
            ]
        ]
    base = symbols(number, parameters(template))
    if number == 62:
        return [dict(base, k_value=k) for k in (1, 2, 3)]
    return [base]


def sample_values(number: int, case: int = 0, path: tuple[int, ...] = ()) -> list[dict]:
    """Values keyed by canonical symbol names; deterministic and candidate-independent."""
    result = []
    for i in range(12):
        r = sp.Rational(i + 2, 7)
        v = {"_default": r}
        if number == 2:
            v.update(
                lambda_plus=sp.Rational(2 + i, 3),
                lambda_minus=sp.Rational(1 + i, 5),
                k_plus=1 + r,
                k_minus=2 + r,
                alpha=1 + r,
                vbar_b=3 + r,
                beta=sp.Rational(i + 1, 13),
                sigma2=sp.Rational(i + 1, 10000),
            )
        elif number == 3:
            v.update(m=1 + r, r0=2 + r, eta=sp.Rational(i + 1, 13))
        elif number == 5:
            v.update(alpha=[0, 1, sp.Rational(1, 2), sp.Rational(1, 5)][i % 4])
        elif number == 7:
            v.update(
                F=sp.Rational((i % 3) + 1, 3),
                k=[0, 1, sp.Rational(2, 3)][i % 3],
                n=1 + i % 4,
                d=1 + (i // 2) % 3,
                q=[sp.Rational(1, 2), 1, sp.Rational(3, 4)][(i // 3) % 3],
            )
        elif number == 11:
            v.update(Delta=r, x=1 - r, K=1 + r, m=1 + i % 4)
        elif number == 15:
            v.update(
                N=8 + i,
                l=1 + i % 5,
                p=[0, 1, sp.Rational(1, 3), sp.Rational(2, 3)][i % 4],
            )
        elif number == 18:
            v.update(
                epsilon0=1 + r,
                k=10 + r,
                z_R=5 + r,
                d0=8 + r,
                alpha_1=1 + r,
                alpha_2=2 + r,
                E_1=3 + r,
                E_2=4 + r,
                phi_1=r,
                phi_2=2 - r,
                m=2 + r,
                Omega_1=1 + r,
                Omega_2=3 + r,
            )
        elif number == 22:
            v.update(x=[0, 1, sp.Rational(1, 2), sp.Rational(1, 3)][i % 4])
        elif number in {23, 24}:
            segment = path[0] if path else 0
            x = (
                -r
                if segment == 0
                else sp.Rational(i + 1, 13) if segment == 1 else 1 + r
            )
            if number == 24 and segment == 1 and i == 0:
                x = sp.Rational(1, 2)
            v.update(
                {
                    "x" if number == 23 else "y": x,
                    "p_z": 2 + r,
                    "mu": 1 + r,
                    "epsilon_UV": sp.Rational(i + 1, 100),
                    "epsilon_IR": -sp.Rational(i + 1, 100),
                }
            )
        elif number == 29:
            v.update(
                {
                    "lambda": 1 + r,
                    "E": 20 + i,
                    "W": 1000 + 50 * i,
                    "alpha": 2 + r,
                    "m": 1 + r,
                    "a_s": (-1) ** i * sp.Rational(i + 1, 100),
                    "hbar": 1,
                }
            )
        elif number == 34:
            v.update(
                a=sp.pi * sp.Rational(i + 1, 30), b=sp.pi * sp.Rational(13 - i, 30)
            )
        elif number == 36:
            v.update(k=1 + r, n=7 + i, X_tot=10 + i)
        elif number == 38:
            v.update(T=1 + r)
        elif number == 39:
            v.update(
                n=[0, 1, 0, 2, 3, 2][i % 6],
                n_prime=[0, 0, 1, 2, 1, 3][i % 6],
                g=1 + r,
                gamma=2 + r,
                alpha=(r + sp.I * (1 - r) if i % 3 else 0),
            )
        elif number == 40:
            v.update(
                chi=1 + r,
                kappa=2 + r,
                sigma=1 + r,
                k=[-2, -1, 0, sp.Rational(1, 2), 1, 2][i % 6],
            )
        elif number == 49:
            v.update(z=4 + i, K=1 + r)
        elif number == 51:
            v.update(x=sp.Rational(i, 2000), g=2 + i % 5, **{"lambda": 1 + r})
        elif number == 53:
            v.update(d=[2, 3, 5, sp.Rational(7, 2)][i % 4])
        elif number == 59:
            v.update(M=10 + 3 * i, epsilon=sp.Rational((-1) ** i, 10000), a=1 + r)
        elif number == 60:
            v.update(Delta_k_sq=2 + r, gamma=sp.Rational(i, 20))
        elif number == 62:
            v.update(phi=sp.pi * sp.Rational(i, 11), k=1 if case == 0 else 2 + i)
        elif number == 66:
            v.update(q=sp.Rational(i, 20))
        elif number == 67:
            v.update(d=2 + i)
        result.append(v)
    return result


def continuation(number: int) -> dict[str, tuple[tuple[sp.Expr, str], ...]]:
    return {
        5: {"alpha": ((sp.Integer(0), "+"), (sp.Integer(1), "-"))},
        7: {"k": ((sp.Integer(0), "+"),)},
        22: {"x": ((sp.Integer(0), "+"), (sp.Integer(1), "-"))},
        24: {"y": ((sp.Rational(1, 2), "+-"),)},
    }.get(number, {})
