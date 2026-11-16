# gradePhyX notice

`formula.py` and the base normalizer in `simplify.py` are adapted from the
MIT-licensed `gradePhyX==1.0.2` wheel published by the PRISM-Physics team.

- Wheel SHA-256: `f01ee559570e8d3fb8f9cfee3a9368d2cac38d8999307f85d1c5c5e729264f70`
- Original formula module SHA-256: `99010c1d425a24ccb3580a61b1bc9f10e86f86d09c01d06f073e76a72f6efb5a`
- Original simplifier SHA-256: `45381b36f773a3fd402760b1676261e48e450ab999b5f5fb5dfa8290facd828f`

DDSR renames the modules and routes LaTeX parsing through
`latex2sympy2-extended`. This avoids the upstream ANTLR 4.11 requirement,
which conflicts with Hydra/OmegaConf's ANTLR 4.9 runtime. The public
comparison API and scoring hyperparameters remain unchanged. The normalizer
also implements the pinned repository's observable script-token handling;
parity is checked against all released solution texts. A misspelled local list
name in the wheel's numeric-unit branch is corrected to prevent a `NameError`;
the official prompt requires numbers outside `\unit{}`, so released scoring is
unaffected.
