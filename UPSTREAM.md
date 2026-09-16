# Upstream provenance

## CritPt

- Repository: https://github.com/CritPt-Benchmark/CritPt
- Commit: `17c2545c302762d2f2d644d923ea4c301605cb08`
- Reused: the prompt text and one-step/two-step conversation behavior.

## Auto_CritPt_Grader

- Repository: https://github.com/CATL-21CLab-SCIAGI/Auto_CritPt_Grader
- Commit: `1787dbb2258af9c606ff97ad6094af5a25e8f246`
- Reused: the deterministic grading contract, with no LLM judge.

## Harbor

- Repository: https://github.com/harbor-framework/harbor
- Pinned package version: `0.22.0`

## SciCode

- Repository: https://github.com/scicode-bench/SciCode
- Commit: `e3158ea011d4235245a547460d3688d7ccbf9900`
- Reimplemented locally: prompts, sequential generation, dataset loading, HDF5
  target decoding, exceptional-step substitutions, and official test execution.

## CMPhysBench

- Repository: https://github.com/CMPhysBench/CMPhysBench
- Commit: `b2cd8571279450f0861759f47d98e9fc577aa993`
- Dataset: `weidawang/CMPhysBench`
- Dataset revision: `43d185851f731e23aa5737c3667b3a9e87bf8cd1`
- Pinned for: the prompt, vLLM sampling parameters (`16384`, `0.6`, `0.95`),
  boxed-answer extraction, and SEED scoring behavior.
- License: Apache-2.0.

## PHYBench

- Repository: https://github.com/phybench-official/phybench
- Commit: `d9db3ec7246f3678aaee65d44a32649ad93beea2`
- Dataset: `Eureka-Lab/PHYBench`
- Dataset revision: `d6d91c787b7abb865eb2490a328bf85a9f5095f0`
- Canonical file: `PHYBench-questions_v1.json` (500 unique problems; 100
  publish reference solutions and answers, and 400 are generation-only).
- Prompt source: arXiv `2504.16074v2`, Appendix D, “Evaluation Experiment
  Setup” (arXiv source file `paper/sec/appendix.tex`).
- Inference settings: provider defaults for API models; local models use
  `temperature=0.6`, `top_p=0.95`, and `max_tokens=32768`.
- Adapted: the official prompt and EED scoring behavior. The repository has no
  inference driver, so the single-user-message envelope and final-balanced-box
  selection are documented project choices.
- License: MIT.
