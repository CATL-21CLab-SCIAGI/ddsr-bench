# Upstream provenance

## CritPt

- Repository: https://github.com/CritPt-Benchmark/CritPt
- Commit: `17c2545c302762d2f2d644d923ea4c301605cb08`
- Reused: the prompt text and one-step/two-step conversation behavior.

### CritPt consensus and long-context generation

- Source: https://github.com/CATL-21CLab-SCIAGI/critpt-eval
- Source commit: `93425280f1829bfced2a51ff52f0d0deba540863`.
- Shared ancestor with DDSR: `aec06ded389df0e60b25b4272c4337269aa4ccda`.
- Integration started from DDSR `2f5b952`. The subsequent consensus refactor
  uses Zhizhou's `a52ca182742555cc3e0c79d5fbe4f50ddc058ca7` on
  `feature/critpt-consensus-migration` as its behavior baseline.
  `codex/critpt-local-submit` descends from that commit, preserving the original
  contributor's commits and authorship rather than squashing or rewriting them.
  Recorded parity checks and their limitations are in
  [consensus testing](ddsr_bench/benchmarks/critpt/evaluation/consensus/TESTING.md).
- Current internal policy: `consensus-61-v2`, 70 slots, 61 active, 9 skipped,
  204 reviewed references. On 2026-09-17, the user approved excluding 47 and 51
  after the [reference audit](ddsr_bench/benchmarks/critpt/evaluation/consensus/SCORING.md). The versioned bundle carries reference provenance
  and template/code digests. This policy is not official CritPt grading.
- External reference bundle (not distributed in Git or wheels):
  `/mnt/workspace/zhizhou/assets/critpt/references/consensus-61-v2.json`.
  Internal submission requires `benchmark.submission.consensus.references`;
  reference diagnostics require `--references` with the local file path. SHA-256:
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
- AI answer-generation prompts: `src/auto_critpt_grader/prompt.py` at
  `565d1b0bfab353e17a40a38fe656c3ab0caee660` (Apache-2.0), adapted into
  `configs/prompts/critpt/ai.yaml`.
  Preserve both system texts and user-message labels. For answers graded by rules,
  instructions reattach the extracted Python template at the end; whitespace
  and template placement can therefore differ from raw source records.
  The loader records `grader: rule` for `grading_plan` records and `grader: llm`
  for `answer_items` records. Prompt routing uses that public field, independently
  of template presence, without accessing private references during generation.
  Empty instructions use the corpus default. Official CritPt prompts are unchanged.

## Harbor

- Repository: https://github.com/harbor-framework/harbor
- Pinned package version: `0.22.0`

## SciCode

- Repository: https://github.com/scicode-bench/SciCode
- Commit: `e3158ea011d4235245a547460d3688d7ccbf9900`
- Dataset: `SciCode1/SciCode`
- Dataset revision: `4510f6a6aa27c43fad7b43da2c59602a86e88480`
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

## CMT-Benchmark

- Dataset: `CMTBenchmark/cmt_benchmark`
- Dataset revision: `c6aa803a6ca553091095ecf07e747b8720b1fbc2`
- Dataset version: `2.0.0` (2026-10-22)
- Scorer repository: https://github.com/JamesRoggeveen/cmt_benchmark_data
- Scorer commit: `c1141831af337724a4a47f66f2410c525c861b66`
- Pinned for: the system instruction, final boxed-answer extraction, numeric and
  parameterized comparison, functional expressions, and non-commuting operator
  normal ordering. The raw v2 `prompt` field remains the user message.
- Verified source shape: 50 records with unique indices 0 through 49; all 50 v2
  references parse and self-score under the pinned upstream scorer.
- Metric policy: the paper and dataset card report strict pass@1 over all 50
  problems. The upstream plotting script instead divides by successfully parsed
  responses. DDSR follows the published fixed denominator, assigns parser
  failures zero reward, and reports parse rate separately.
- The maintained v2 dataset supersedes the anonymous dataset still named in the
  pinned repository's query configuration. DDSR uses the maintained dataset and
  does not incorporate upstream model clients, plotting, or dataset-build scripts.
- License boundary: the Hugging Face dataset and repository README declare MIT,
  but the scorer commit has no standalone license file or notice. Until that is
  clarified, DDSR will not copy the scorer source; it will implement the public
  behavior and check it differentially against the pinned repository.

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

## PRISM-Physics

- Repository: https://github.com/Open-PRISM/PRISM-Physics-Code
- Repository commit: `d2958d12a99e7a38a41832e4e0cc9f40ef7d474f`
- Paper: https://arxiv.org/abs/2510.03185
- Project site: https://open-prism.github.io/PRISM-Physics/
- Dataset: `open-prism/PRISM-Physics`
- Dataset revision: `89d1e1cb91ec14b319482ec36b43fca5305f9c86`
- Dataset files: `01_cleaned_dag.json` through `07_cleaned_dag.json` (1,401
  records; raw IDs repeat between shards, so DDSR uses `shard-id`).
- Prompt source: `utils/prompt_utils.py:get_eval_prompt` at the pinned repository
  commit. Inference defaults are also reported in arXiv `2510.03185v2`,
  Appendix E.1.
- Formula engine: `gradePhyX==1.0.2`, PyPI wheel SHA-256
  `f01ee559570e8d3fb8f9cfee3a9368d2cac38d8999307f85d1c5c5e729264f70`.
- Pinned for: the text-only inference prompt, output settings (`8096`, `0.0`),
  low effort for supported reasoning models, formula extraction and matching,
  ancestor-closure DAG score, and correctness threshold (`0.8`).
- Dataset and formula engine licenses: MIT. The benchmark code repository has
  no standalone license file. DDSR reproduces the evaluation prompt required
  by the benchmark contract and reimplements the public orchestration, but
  vendors no Python code from that repository; the formula engine is copied
  only from its separately MIT-licensed wheel.
- DDSR vendors the engine's formula-comparison and base normalization modules
  under their MIT license, with a compatibility layer for the pinned repository's
  script-token normalization. Its `sympy.parse_latex` call is routed through
  `latex2sympy2-extended`: gradePhyX requires ANTLR 4.11, while the framework's
  Hydra/OmegaConf runtime requires ANTLR 4.9. Algebraic rearrangement, unit
  conversion, and inverted-equation cases are checked against the pinned
  repository behavior.
- The dataset contains image metadata, but its Hugging Face repository does not
  distribute those referenced assets. DDSR initially exposes the official
  text-only setting and preserves image metadata for a later multimodal source.
