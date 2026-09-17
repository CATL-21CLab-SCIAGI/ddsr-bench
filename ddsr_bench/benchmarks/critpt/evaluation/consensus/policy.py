"""Reviewed scope and precision policies; never inferred from candidate output."""

from __future__ import annotations

from dataclasses import dataclass

SKIP = {6, 12, 19, 25, 30, 33, 47, 51, 68}
SKIP_REASONS = {
    47: (
        "Excluded after the 2026-09-17 audit: theta=x violates the stated constant "
        "boundary condition and the full trace diverges. All four flagship reference "
        "values (~11.891648076711) match the configuration with theta and phi swapped."
    ),
    51: (
        "Excluded after the 2026-09-17 audit: returns to the origin require even time "
        "steps, but all four flagship references give Z(3)=2g. Independent path "
        "enumeration contradicts the references; the elementary-function requirement "
        "also conflicts with the elliptic-integral case g=2, lambda=1."
    ),
}
MODES = {
    "symbolic": {3, 11, 15, 18, 29, 34, 36, 38, 39, 53, 60, 67},
    "set": {4, 13, 40, 54, 58, 65},
    "domain": {5, 7, 22, 23, 24, 62},
    "series": {2, 49, 59, 66},
    "function": {20, 45},
}
MODES["value"] = set(range(1, 71)) - SKIP - set.union(*MODES.values())


@dataclass(frozen=True)
class Precision:
    relative: float = 0.01
    absolute: float = 0.0
    exact: bool = False


def mode(number: int) -> str:
    if number not in range(1, 71):
        raise ValueError("challenge must be between 1 and 70")
    return next((k for k, v in MODES.items() if number in v), "skip")


def precision(number: int, path: tuple[int, ...]) -> Precision:
    """Internal half-last-place acceptance, not an official hidden tolerance."""
    first = path[0] if path else None
    if number in {4, 28, 43, 50, 54}:
        return Precision(exact=True)
    if number == 13 and path and path[-1] < 4:
        return Precision(exact=True)
    if (number == 44 and first in (0, 1)) or (number == 61 and first == 0):
        return Precision(exact=True)
    absolute = {
        14: 0.0005,
        17: 0.00005,
        37: 0.00005,
        44: 0.0005,
        46: 0.00005,
        47: 0.0000005,
        48: 0.000000005,
        52: 0.0005,
        55: 0.0005,
        61: 0.005,
        63: 0.0005,
    }
    if number in absolute:
        return Precision(0.0, absolute[number])
    if number == 31:
        return Precision(0.0, (0.5, 0.02)[first])
    if number == 32 and first in (0, 1):
        return Precision(0.0, (0.05, 0.02)[first])
    if number == 35:
        return Precision(0.01, 1e-9)
    if number in {20}:
        return Precision(1e-6, 0.0)
    return Precision()


SIGNIFICANT_DIGITS = {26: 3, 27: 3, 41: 2, 56: 3}

NOTES = {
    4: "Only the Fable/GPT tied group is available; other top group is missing.",
    15: "Positive integer N,l; noise p in [0,1]. Finite checks use interior chain lengths.",
    20: "nm/mW/SI template units; nondegenerate prolate, stable trapping fixtures. Coverage is finite; epsilon_r>1 is a fixture choice, not a universal restriction.",
    23: "p0=pz and positive-energy external state imply pz>0. Separate x domains; Laurent poles retained.",
    24: "Separate y domains and principal analytic branches; finite continuation at y=1/2.",
    29: "Trusted reference V0=alpha*E^2 plus deep positive well implies alpha>0; scattering length may have either sign.",
    35: "225 ordered coefficients, Y0Y1=+1. Matching does not certify commutators or eigenstate constraints.",
    36: "Trusted reference threshold n_min=7; moment checked at n>=7, threshold checked separately.",
    37: "Historical 1% consensus retained; each available complete reference is accepted at field precision.",
    45: "Positive masses follow conduction-minimum / valence-maximum dispersion. Includes equal ratios returning False.",
    49: "Compare requested large-z powers and logarithms; omit z-independent terms.",
    53: "Reported Fable source duplicates GPT; reported confidence is not recomputed.",
    62: "k_value selects k=1 or k>1, while symbolic dependence on k is preserved.",
    65: "Canonical formal tr/psi syntax only; no general fermion algebra or numeric substitution.",
    **SKIP_REASONS,
}
