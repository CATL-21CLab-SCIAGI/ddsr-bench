# Upstream provenance

## CritPt

- Repository: https://github.com/CritPt-Benchmark/CritPt
- Data commit: `bc37a22bedb8744dff8996ccce7f9eedec548915` (CritPt v2).
- Dataset revision: `37ecba7e9d00eda858d7be1adba19a38ba50e7a6`.
- Prompt commit: `17c2545c302762d2f2d644d923ea4c301605cb08`.
- Reused: v2 challenge data plus the inspected prompt text and one-step/two-step
  conversation behavior. Data and prompt revisions are intentionally separate.
- Data status: official, revised. The v2 commit changes 52 of 70 challenge JSON
  records relative to the prompt commit, but publishes no item-level correction
  ledger; DDSR therefore does not describe every change as a verified fix.

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

- Corrected repository: https://github.com/flyingwagner/scicode-verified
- Repository commit: `a7b0a308b99cb010e51bfdee535059c3ebe5e8b0`
- Dataset: `shhu2001/SciCode-Verified`
- Dataset revision: `752eb6b786b0e0d0a3b44cdb7c066c559b952658`
- Dataset version: `v2.2` (64 problems; 287 scored subproblems).
- Problem MD5: `e107a70b4de390c25852348c19567d33`; matching HDF5 MD5:
  `32c5602f2bf08393f917f03ff0b80d40`.
- Data status: independent, corrected and human-verified. The release excludes
  one underdetermined original problem and retains 64 problems with 287 scored
  subproblems.
- Reimplemented locally: prompts, sequential generation, dataset loading, HDF5
  target decoding, exceptional-step substitutions, and SciCode-compatible test
  execution.

## CMPhysBench

- Repository: https://github.com/CMPhysBench/CMPhysBench
- Commit: `b2cd8571279450f0861759f47d98e9fc577aa993`
- Dataset: `weidawang/CMPhysBench`
- Dataset revision: `43d185851f731e23aa5737c3667b3a9e87bf8cd1`
- Pinned for: the prompt, vLLM sampling parameters (`16384`, `0.6`, `0.95`),
  boxed-answer extraction, and SEED scoring behavior.
- License: Apache-2.0.
- Data status: official, unverified. No accessible semantic correction was found
  in the bounded search recorded in [DATASETS.md](DATASETS.md).

## CMT-Benchmark

- Dataset: `CMTBenchmark/cmt_benchmark`
- Dataset revision: `c6aa803a6ca553091095ecf07e747b8720b1fbc2`
- Dataset version: `2.0.0` (2026-10-22)
- Data status: official, corrected. Its changelog records expert-reviewed fixes
  to 10 problems, including seven changed reference answers.
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
- Corrected data source: https://github.com/Science-Discovery/LOCA
- Corrected data commit: `0e3b86b2ee18ab6118dd57029a3f8de444db365e`.
- Corrected file: `problem_set/PHYBENCH_100/PHYBENCH_100_fixed.jsonl` (100
  public-reference problems), SHA-256
  `262a0037b4339a0ab5ce330a7b083c4bd2513399ac952aa7805999e16fe7d7f3`.
- Data status: independent, corrected. LOCA section 4.2 reports expert review,
  20 incorrect PHYBench answers, and corrections where possible; this is not an
  official PHYBench revision or a claim of conclusive verification of all rows.
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
- Data status: official, unverified. The accessible Alliance release is a
  74-problem audited subset of a frozen 100-problem sample, not a correction of
  the full 1,401-record formula-DAG dataset.
- DDSR vendors the engine's formula-comparison and base normalization modules
  under their MIT license, with a compatibility layer for the pinned repository's
  script-token normalization. Its `sympy.parse_latex` call is routed through
  `latex2sympy2-extended`: gradePhyX requires ANTLR 4.11, while the framework's
  Hydra/OmegaConf runtime requires ANTLR 4.9. Algebraic rearrangement, unit
  conversion, and inverted-equation cases are checked against the pinned
  repository behavior.
- The upstream driver defaults to text mode and offers a separate multimodal mode
  whose image base defaults to an unpublished local `PHYBE` tree. The pinned
  Hugging Face tree bundles no JPEG assets, although its JSON contains 90 public
  CDN URLs, 596 unresolved root-relative image paths, and one malformed HTML
  location. DDSR keeps this released metadata in its raw loader but omits unused
  image metadata from text-only tasks and generation records. Four question
  texts contain embedded relative Markdown image links; DDSR retains those
  upstream prompt strings and documents the unresolved assets rather than
  silently rewriting the questions.

## PHYSICS

- Original benchmark: https://github.com/yale-nlp/Physics
- Verified dataset: `yale-nlp/physics-verified`
- Dataset revision: `724170025e3541c9b877285b2d577a1257796f0b`.
- Dataset tree used by DDSR: `data/validation.jsonl` and `data/test.jsonl`, tree
  SHA-256 `06edf8cfbf0fbfc81652033b35bdfedc021a1eac14104823f83910b9d777a5e9`.
- Evaluation harness: https://github.com/yale-nlp/physics-verified
- Harness commit: `3b36e33cd3a904b86dd9b95d8e8ca4b49e81ce4b`.
- Data status: official, verified. The original PHYSICS repository recommends
  PHYSICS-Verified for new evaluations and retains the original release as the
  paper artifact.
- Verified shape: 240 validation problems/620 answers and 794 test
  problems/1,939 answers. There are 248 image-bearing problems with 265 embedded
  PNG data URIs; validation and test IDs are disjoint.
- Pinned for: the multimodal message order, response format, per-problem judge
  prompt, one verdict per reference answer, malformed-judge retry behavior, 1%
  numeric relative tolerance, strict accuracy, answer accuracy, partial credit,
  HLE RMS calibration error, a 3,600-second request timeout, and seeded
  2,000-repeat bootstrap intervals. The upstream CLI defaults to 16 concurrent
  requests; the bundled DDSR jobs use four as a conservative operational choice.
- Exact prompt SHA-256 values: response format
  `17ffc9761eb3a67a475f4e2e036f9a5a0fe05d864a6b7e86c52c4589f89d738e`;
  judge `fb9a31a4242b0422056967125892ce694d2eb0030aa71bf935cfcf1bcea85c7f`.
- Published generation settings are checkpoint-specific: thinking models use up
  to 65,536 output tokens, non-thinking models up to 8,192, and GPT models use
  high reasoning effort. The published judge is GLM-5.3-Flash served locally;
  scores from different judges are not directly comparable.
- Project choice: DDSR uses shared provider clients and records their explicit
  request settings. Prepared tasks are a static/offline fallback and do not
  introduce a container verifier because official scoring requires an LLM judge.
  Shared DDSR resume keeps terminal failures; the upstream append-only runner
  instead retries failed generation or missing judgments when rerun.
- Dataset and harness license: MIT. The upstream notice is distributed beside
  the copied prompt text at `configs/prompts/physics/LICENSE.txt`.

## HLE-Verified

- Original benchmark and evaluator: https://github.com/centerforaisafety/hle
- Evaluator commit: `22ed3074b1e7b134bcbc09028d0ba320839b0655`.
- Verified release: https://github.com/SKYLENAGE-AI/HLE-Verified
- Verified repository commit: `b705e0fb541c025a1532ce0d60d70ae2f53b00e0`.
- Paper: https://arxiv.org/abs/2602.13964
- Dataset: `skylenage-ai/HLE-Verified`, revision
  `0bc83643672d4f68a5f89998617a639d85e7318b`.
- Dataset tree: 15 Parquet shards, SHA-256
  `5dc045c3eb1629de5ed7bcfb01cb95a09fac0295b00d2f4e614696b07ba1d27c`.
- Data status: official derivative, verified. The source contains 668 Gold,
  1,143 Revision, and 689 Uncertain records. DDSR's default certified selection
  is Gold+Revision (1,811), while source validation covers all 2,500.
- Pinned for: exact response-format and judge prompts, multimodal message order,
  the o1 role exception, structured judge fields, default judge
  `o3-mini-2025-01-31`, 4,096-token judge cap, fixed-denominator accuracy,
  Wald 95% half-width, and HLE RMS calibration error including its skipped last
  bin. The original clients use 600-second generation and 300-second judge
  timeouts with one transport retry. Generation leaves checkpoint decoding
  defaults unchanged. HLE-Verified uses five independent rollouts and reports
  avg5 accuracy; DDSR keeps the original ten-request concurrency default.
- Exact prompt file SHA-256 values: response format
  `cd27d28bdfe308dab9c78659fb0ff24dee3d85cdf0340ec4e37f984c5fcf1aac`;
  judge `732dd73cab327aca3bc0d6357d0c2254c0fdeeb8928d2c51edc51648d1be4fb8`.
- Project choice: prepared tasks are an offline fallback; scoring stays on the
  host because no model code is executed. Non-OpenAI job profiles substitute a
  provider judge and therefore do not claim official-judge score parity.
- License boundary: original HLE evaluation code and copied prompts are MIT,
  and the HLE-Verified paper is CC BY 4.0. No explicit license was found for the
  HLE-Verified repository or dataset at the pinned revisions; DDSR does not
  redistribute its data. The HLE MIT notice is copied beside the prompts.

## FrontierScience

- Dataset: `openai/frontierscience`, revision
  `25ed67db7da8f4591484e764008ff585544f5a30`.
- Dataset tree: `olympiad/test.jsonl` and `research/test.jsonl`, SHA-256
  `a3b5bd97aff5aabb76a3ee0a7d1ab6cc87b0306d44ef21bb26bef16b26816d2a`.
- Official paper: https://arxiv.org/abs/2601.21165.
- Maintained Inspect implementation: https://github.com/UKGovernmentBEIS/inspect_evals,
  release commit `a4205c238a40de36204c487fa38d7720bc1085cf`.
- Pinned for: the verbatim problem input, official Olympiad equivalence and
  Research rubric prompts, terminal judge markers, ten-point normalization,
  seven-point Research success threshold, GPT-5 high-effort judge setting, and
  empty-response behavior. The paper averages 20 Olympiad and 30 Research
  trials; DDSR leaves the job attempt count explicit so the tracks can be run
  separately with those settings.
- The maintained Inspect scorer calls this format `olympic`, while the official
  dataset path and paper terminology use Olympiad. DDSR uses `olympiad` for the
  data-facing track name and preserves the scoring behavior.
- Data status and license: official, verified gold set; Apache-2.0. The dataset
  card requests that the released evaluation data not enter training corpora.
  DDSR does not redistribute raw records and keeps references out of exported
  model trajectories.

## OlympiadBench

- Repository: https://github.com/OpenBMB/OlympiadBench, commit
  `ba5b26a7e2849940b598a9159c1190daa2b9175f`.
- Dataset: `Hothan/OlympiadBench`, revision
  `91184b52131e7fc9455fef848035173aea8cc01a`.
- Dataset tree: 18 Parquet files below `OlympiadBench/`, materialized-tree
  SHA-256 `24d949234fafd2562d81a4701b96ee21632e8af6dcd1b393889dc4b151868d30`.
- Pinned for: bilingual system prompts; answer-type, multiple-answer, unit, and
  theorem-proof instructions; physics context ordering; multimodal placeholder
  ordering; deterministic 2,048-token generation settings; boxed extraction;
  percentage, numeric, expression, equation, interval, and unordered-answer
  comparison.
- The current dataset uses `<image_n>` placeholders while the older GPT-4V
  helper in the repository splits `<img_n>`. DDSR follows the released data
  placeholders and verifies a one-to-one match with embedded assets.
- The current `eval/auto_scoring_judge.py` documents its first argument as the
  ground truth. DDSR follows that order rather than the reversed call in the
  legacy `inference/judge.py`. `latex2sympy2-extended` replaces only the parser
  runtime dependency, not the comparison policy.
- Official automated scoring covers open-ended problems only. Theorem proofs
  were sampled for human evaluation; DDSR records them as ungraded.
- License: MIT.

## HiPhO

- Official repository: https://github.com/SciYu/HiPhO, commit
  `8db13ae9aa6e3d4ddb7bb63c8f6a568d5b207243`.
- Dataset: `SciYu/HiPhO`, revision
  `8e196c09a71e4e68b75c422defa512473359e0e5`.
- Paper: https://proceedings.mlr.press/v306/yu26ay.html.
- Current source: 403 subquestions across 14 exams and 131 PNG assets; pinned
  tree SHA-256 `47f0041d39e57d9e7ec28669b21cd0cdc0218549719a59a90509a0c0e4ceeec8`.
- Published cohort: 360 subquestions across 13 exams. CPhO 2025 is a later
  official extension and is excluded when `cohort: published` is selected.
- Maintained evaluator: `open-compass/VLMEvalKit`, commit
  `ed3e918edf4a62e0bf42a2768655e02d45312fef`. The three vendored files are
  byte-identical to the reproducible `bytedance/axion1_0` wrapper at commit
  `96ce1830d87399f20577790f405fe2407da34156`.
- Pinned for: exact English/Chinese inference prompts, image-first message
  order, answer extraction and equivalence, Gemini-2.5-Flash marking-criterion
  grading, final `max(answer_points, step_points)` policy, and arithmetic mean
  of per-exam raw point totals. Published inference uses temperature `0.6`, the
  largest output limit available per model, and eight independent attempts.
- Project choice: VLMEvalKit predates CPhO support and marks only
  PanMechanics as Chinese. DDSR also routes CPhO 2025 through the unchanged
  Chinese template because the released exam text and data sheet are Chinese.
  Static evaluation is primary because no generated program is executed.
- Data license: MIT. Vendored evaluator and shim: Apache-2.0; notices and exact
  file digests are distributed beside the code.
