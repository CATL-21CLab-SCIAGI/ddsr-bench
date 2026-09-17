# Upstream provenance

## CritPt

- Repository: https://github.com/CritPt-Benchmark/CritPt
- Commit: `17c2545c302762d2f2d644d923ea4c301605cb08`
- Reused: the prompt text and one-step/two-step conversation behavior.

### CritPt consensus and long-context migration

- Source: https://github.com/CATL-21CLab-SCIAGI/critpt-eval
- Source commit: `93425280f1829bfced2a51ff52f0d0deba540863`.
- Shared ancestor with DDSR: `aec06ded389df0e60b25b4272c4337269aa4ccda`.
- Current internal policy: `consensus-61-v2`, 70 slots, 61 active, 9 skipped,
  204 reviewed references. On 2026-09-17, the user approved excluding 47 and 51
  after the [reference audit](ddsr_bench/benchmarks/critpt/evaluation/consensus/EXCLUSIONS.md). The versioned bundle carries reference provenance
  and template/code digests. This policy is not official CritPt grading.
- External reference bundle (not distributed in Git or wheels):
  `/mnt/workspace/zhizhou/assets/critpt/references/consensus-61-v2.json`.
  Use `--bundle` for other mounts or installations. SHA-256:
  `6b9edc5057a92148701bed69afa3b4fb121da89223c6978dc01ddca7005a44bc`.
  Historical `consensus-63-v1` had 63 active problems and 212 references; its
  hash was `fa0facc0082261b6f42d976e07b8bc1c9302eff517e7802c9bf49f87fd4a3f2f`.
- Original code validator from `d83b0d9`, unchanged in DDSR:
  `1b388bfd2107206393c9c121b8dcfb25065921880f80fa7b9c52a129e4bb5a50`.
- Migrated project choices: concurrency slot refill, streaming journals,
  atomic stage checkpoints, deterministic trial-seed derivation, dynamic vLLM
  context budgeting and Linux worker isolation. Prompt and fallback behavior
  remain pinned upstream behavior; these operational additions are local.
- Historical local inference: Qwen3.8-27B, xhigh, native context 262144 without
  YaRN, 32 concurrent trials; DeepSeek v4 Flash 0731, max reasoning, PAI Chat
  Completions, 16 concurrent trials, first-stage cap 393216. Both used five
  attempts, seed base 42 and an initial formatting cap 65536; selected retries
  used 131072 with unchanged per-trial seeds. These are recorded run settings,
  not a claim about unpublished Artificial Analysis request bodies.
- License: the source CritPt integration does not ship a standalone license
  file; migration preserves existing provenance and does not grant a new
  license to reference material. Reference distribution remains subject to
  the source repositories' terms.

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
