# Exclusions added in consensus-61-v2

On 2026-09-17, the user approved excluding Challenges 47 and 51 after comparing
the original statements with every available flagship reference. This is an
internal scoring policy change, not a claim about official CritPt grading.
There are now 61 active main problems and 9 skipped problems. The 70 problem
slots and every attempt remain present; skipped results carry `matched: null`
and a reason, and contribute to neither numerator nor denominator.

## Challenge 47: reference values correspond to swapped angles

The pinned statement specifies `theta=x`, `phi=(2*pi/3)*exp(-x*x)` and a
constant spin at infinity. Its `m_z=cos(x)` violates that boundary condition;
the full-space trace diverges even if the boundary assumption is relaxed.

Opus, Fable and Kimi return 11.8916480767; GPT returns 11.891648076712.
Opus's code comment instead describes a Gaussian polar angle and linear
azimuthal angle. Independently swapping the two angles reproduces
11.891648076711 at 128, 256 and 512 mapped Gauss–Legendre nodes. Thus the
finite consensus reference is unsuitable for the literal statement.

The divergent background has trace density 16/pi, which is not the full trace.
Exclusion does not accept every infinity/NaN answer or recover a submission
from first-stage reasoning.

## Challenge 51: consensus contradicts return-path parity

The stated walks, splitting and recombination require even time to return to
the origin. All four references nevertheless expand as
`1 + (g + 2*lambda**2)*x**2 + 2*g*x**3 + O(x**4)`.
The nonzero `Z(3)=2g` contradicts the original rules for every allowed `g>=2`.

For `g=2, lambda=1`, independently enumerated `Z(0..8)` is
`[1,0,4,0,36,0,400,0,4900]`; all four references instead give
`[1,0,4,4,32,68,336,984,4096]`. The elementary-function requirement also needs
review: this case gives `2*K(4*x)/pi`, a complete elliptic integral (modulus
convention). DeepSeek's integral submissions agree with the path recurrence
and independent enumeration, while two Qwen submissions match the defective
reference. Exclusion removes both potential false negatives and false positives.

## Provenance and scope

- Original problem commit: `17c2545c302762d2f2d644d923ea4c301605cb08`.
- Previous policy: `consensus-63-v1`; original bundle SHA-256:
  `fa0facc0082261b6f42d976e07b8bc1c9302eff517e7802c9bf49f87fd4a3f2f`.
- Audit source commit: `f3af5a674124915ca19eee1ac399ca1538ac71ef`.
- Local evidence, code and per-attempt findings:
  `/mnt/workspace/zhizhou/assets/critpt/audits/false-negative-20260917/README.md`.
- The remaining 61 problems retain their reference code, comparison policies,
  confidence and original generation validation. Challenge 45 remains active
  pending further review. No general format restrictions were relaxed.
- Existing v1 reports remain historical. Revised reports must identify v2 and
  its bundle hash; a five-attempt mean uses 305 active trials per model.
