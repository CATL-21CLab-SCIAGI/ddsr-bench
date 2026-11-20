# Dataset editions

DDSR-Bench pins a named edition of every benchmark. `official` describes who
published an edition; it does not imply that the data are defect-free.

| Status | Meaning |
| --- | --- |
| `revised` | Upstream published a new edition with substantive data changes but no correction ledger sufficient to classify every change. |
| `corrected` | Problems or references were repaired while preserving the intended benchmark where feasible. |
| `verified` | A reviewed or filtered subset; its population may differ from the original benchmark. |
| `audited` | An independent expert-audit release, which may use a different subset or evaluator. |
| `parser-repaired` | Reference formatting was changed for machine parsing; semantic correctness was not established. |
| `unverified` | No accessible corrected edition was found. |

Source authority and status are recorded separately. For example, CMT v2 is an
official corrected release, whereas the Alliance CMT edition is independently
audited and disputed by the benchmark authors.

## Selected editions

| Benchmark | Selected edition | Authority | Status | Scope |
| --- | --- | --- | --- | --- |
| CritPt | v2 at `bc37a22bedb8744dff8996ccce7f9eedec548915` | Official | `revised` | 70 challenges |
| SciCode | SciCode-Verified v2.2 at `752eb6b786b0e0d0a3b44cdb7c066c559b952658` | Independent audit | `corrected` | 64 problems, 287 scored subproblems |
| CMT-Benchmark | v2.0.0 at `c6aa803a6ca553091095ecf07e747b8720b1fbc2` | Official | `corrected` | 50 problems |
| PHYBench | LOCA correction at `0e3b86b2ee18ab6118dd57029a3f8de444db365e` | Independent correction | `corrected` | 100 public-reference problems |
| PRISM-Physics | `89d1e1cb91ec14b319482ec36b43fca5305f9c86` | Official | `unverified` | 1,401 problems |
| CMPhysBench | `43d185851f731e23aa5737c3667b3a9e87bf8cd1` | Official | `unverified` | 520 problems; 100 locally gradable |
| PHYSICS | PHYSICS-Verified at `724170025e3541c9b877285b2d577a1257796f0b` | Official | `verified` | 1,034 problems, 2,559 scored answers |
| HLE | HLE-Verified at `0bc83643672d4f68a5f89998617a639d85e7318b` | Official derivative | `verified` | 2,500 source items; 1,811 Gold+Revision selected |
| FrontierScience | `openai/frontierscience` at `25ed67db7da8f4591484e764008ff585544f5a30` | Official | `verified` | 160 gold-set items across two tracks |
| GPQA | `idavidrein/gpqa` at `56686c06f5e19865c153de0fdb11be3890014df7` | Official | `original` | 448 Main, 198 Diamond, 546 Extended questions |
| MMLU | Official `data.tar` with code at `4450500f923c49f1fb1dd3d99108a0bd9717b660` | Official | `unverified` | 14,042 test questions across 57 subjects |

CritPt prompts remain pinned independently to the inspected official prompt
commit recorded in [UPSTREAM.md](UPSTREAM.md). A data correction does not
silently change the model-elicitation contract.

## Local materialization

Downloaded source files live under the ignored path
`datasets/<benchmark>/<edition>/`; prepared tasks mirror that layout at
`tasks/<benchmark>/<edition>/`.
Edition names identify the source or major version (`loca`, `v2`, `v2.2`), while
the exact release, revision, authority, and correction status stay in benchmark
configuration and this catalog. Pass the local file or shard directory through
`paths.input` during `action=prepare`.

Materialize all selected editions, or one edition, with the shared catalog:

```bash
ddsr-bench action=fetch benchmark=all
ddsr-bench action=fetch benchmark=phybench
```

The catalog owns acquisition recipes; existing benchmark loaders own semantic
validation. Downloads are staged outside the published edition, checked against
the configured tree SHA-256 and native loader, then atomically published. An
exact edition is reused; an invalid existing edition causes an error and is not
replaced. Raw sources and prepared tasks remain uncommitted artifacts.

## Evidence and alternatives

### CritPt and CMT-Benchmark

- The official [CritPt v2 commit](https://github.com/CritPt-Benchmark/CritPt/commit/bc37a22bedb8744dff8996ccce7f9eedec548915)
  changes 52 of the 70 challenge JSON records relative to the inspected v1
  commit. Upstream identifies this as the deployed v2 release but provides no
  item-level correction ledger, so DDSR records it as `revised` rather than
  inferring that every change is a correction.
- The official [CMT-Benchmark v2.0.0 changelog](https://huggingface.co/datasets/CMTBenchmark/cmt_benchmark)
  documents expert-reviewed fixes to 10 problems, including seven changed
  reference answers, without adding or removing problems. This supports the
  `corrected` classification directly.

### SciCode

- [SciCode-Verified](https://huggingface.co/datasets/shhu2001/SciCode-Verified),
  accompanied by its [audit paper](https://arxiv.org/abs/2608.04975),
  is the only accessible semantic correction found. Its audit reports 263
  defects across the original 65-problem test split, corrects every confirmable
  defect, and excludes one underdetermined problem. Its v2.2 problem file has
  SHA-256 `542b368f3f17fcb92c6159907c10f8e3ca050681d2c826188ae7e3dfc6726b8d`;
  the release manifest records MD5 `e107a70b4de390c25852348c19567d33`
  for that file and `32c5602f2bf08393f917f03ff0b80d40` for the HDF5 targets.
- `SciCodePile/SciCode-Runnable-Benchmark-Reviewed` is a different runnable-code
  corpus, not a correction of the original SciCode evaluation set.

### PHYBench

- The [LOCA paper](https://arxiv.org/abs/2510.01249) section 4.2 says its
  structured outputs enabled human experts to review the 100 public-reference
  problems, manually identify 20 incorrect answers, and provide corrections
  where possible. The pinned
  [LOCA release](https://github.com/Science-Discovery/LOCA) contains the
  resulting `PHYBENCH_100_fixed.jsonl`; its SHA-256 is
  `262a0037b4339a0ab5ce330a7b083c4bd2513399ac952aa7805999e16fe7d7f3`.
- This is the strongest accessible full semantic correction found, but it is an
  independent release, not an official PHYBench revision or a claim that every
  record has been conclusively re-verified.
- A local compatibility probe of the pinned data and EED code found that all
  100 references self-score through EED's identical-string shortcut, while 77
  still score 100 after trailing whitespace bypasses that shortcut. The other
  23 expose an EED parsing limitation; LOCA's semantic corrections do not claim
  to repair the benchmark metric.
- LOCA also publishes a 59-problem accepted subset. It is `verified`, not the
  full corrected benchmark. SHA-256:
  `c7f748ca1d5b815e9230d5056a43fe3e7a0ceda5115c180efaf8368e13224175`.
- [PhyBench_fixed](https://huggingface.co/datasets/PKUfudawei/PhyBench_fixed)
  revision `7ad9c3c21db4fe0d19bcf2cbe5afc84565a2f27f` repairs 22 references so
  SymPy can parse them. It is `parser-repaired`, not a semantic correction.
- The [Alliance audit](https://github.com/Allliance/physics-benchmarks) retains
  87 of the 100 public-reference problems and scores them with its shared judge.
  It is an `audited` subset, not interchangeable with native EED evaluation.

### PRISM-Physics, CMT-Benchmark, and CritPt

The Alliance release at `2e081374d9597171d68380e2e3b5b83d9cc9d446`
publishes independently audited subsets: 74 PRISM problems, 49 CMT problems,
and 54 CritPt problems. It uses one HLE-derived judge across benchmarks, so DDSR
does not route those records through the benchmarks' native scorers.

| Alliance post-audit file | SHA-256 |
| --- | --- |
| PHYBench (87) | `d231c1e4b66cd677ae139ceee3c9aa9ccd792ac545f2f74a3267ebcfcb17e3eb` |
| PRISM-Physics (74) | `1c7ddc6a29f3f64924278a08c35f669280d16d68f39b2cbf2412e559ee7534ac` |
| CMT-Benchmark (49) | `166428c858d4a71efae870599f95b6c2497205f81c9a0643dc4c72715c7100ec` |
| CritPt (54) | `8fb4a844c64eb2d6709c116bad459ae5a87533c51a5059493848784291a9d4c9` |

For CMT, the selected official v2 release documents its own corrections and the
authors' disagreement with most Alliance changes. Both editions remain named;
neither is silently substituted for the other.

No accessible full semantic correction was found for PRISM-Physics. Its
Alliance edition covers a frozen 100-problem text-only subset, not the official
1,401-problem formula-DAG dataset.

The official JSON records include 687 image locations, but the pinned Hugging
Face tree bundles no JPEG files. Ninety locations are absolute public CDN URLs;
596 are unresolved root-relative paths, and one is malformed HTML. Upstream
multimodal evaluation expects a separate local image tree, so these URLs do not
form a complete multimodal release. DDSR's official edition follows the upstream
default text mode and excludes unused image metadata from normalized tasks. Four
question texts embed relative image links directly; they remain unchanged for
source parity, although their assets were not found in the checked public trees.

### CMPhysBench

The public dataset has 520 records, but only 100 contain reference answers. A
search of the official repository, issues, forks, Hugging Face datasets, and
paper-linked resources found no accessible corrected edition. The
`budecosystem/cmphysbench` dataset identifies itself as a mirror.

### PHYSICS

- The original [PHYSICS repository](https://github.com/yale-nlp/Physics)
  recommends PHYSICS-Verified for new evaluations. The official audited edition
  removes proof-, explanation-, drawing-, and unavailable-figure-only requests;
  independently resolves reference disagreements; and separates each requested
  result for answer-level scoring.
- The pinned release contains 240 validation problems with 620 answers and 794
  test problems with 1,939 answers. Its 248 image-bearing problems embed 265 PNG
  data URIs, so the selected edition is self-contained.
- PHYSICS-Verified is a derivative edition under benchmark key `physics`, not a
  separate DDSR benchmark name. The original 1,297-problem PHYSICS artifact is
  retained upstream but is not selected for new evaluation.

### HLE

- [HLE-Verified](https://huggingface.co/datasets/skylenage-ai/HLE-Verified)
  audits all 2,500 HLE records into 668 Gold, 1,143 Revision, and 689 Uncertain
  items. The accompanying paper recommends Gold for stable leaderboards,
  Revision for robustness, and Uncertain for ambiguity research.
- DDSR's `verified` edition downloads all three classes but selects Gold and
  Revision by default. This 1,811-item certified population is an explicit DDSR
  policy, not a claim that the HLE-Verified paper's “Full Set” excludes
  Uncertain records.
- The selected population has 1,372 exact-match and 439 multiple-choice items,
  including 211 embedded question images. The complete source has 342 embedded
  question images. DDSR also preserves the 376 released rationale images
  verifier-side, 272 of which belong to the selected population.
- The pinned dataset tree SHA-256 is
  `5dc045c3eb1629de5ed7bcfb01cb95a09fac0295b00d2f4e614696b07ba1d27c`.
  No explicit dataset license was found at the pinned repository or dataset
  revision, so raw data remain downloaded, ignored local artifacts.

### FrontierScience

- The official Apache-2.0 Hugging Face release contains 100 Olympiad and 60
  Research problems. The source tree digest is
  `a3b5bd97aff5aabb76a3ee0a7d1ab6cc87b0306d44ef21bb26bef16b26816d2a`.
- The public set has 70 physics, 60 chemistry, and 30 biology items. DDSR
  validates those counts and the globally unique task-group IDs before use.
- The release calls itself a gold set and asks that it not be used for model
  training. DDSR supports evaluation and can export trajectories for auditing;
  users are responsible for honoring the source's training-data restriction.

### OlympiadBench

- The pinned official revised release contains 8,476 problems across 18 Parquet
  subsets: 6,142 mathematics, 2,334 physics, 6,351 Chinese, 2,125 English,
  6,728 open-ended, and 1,748 theorem-proof records. Its 4,869 multimodal
  problems contain 5,875 embedded JPEG/PNG assets.
- The materialized 18-file source tree SHA-256 is
  `24d949234fafd2562d81a4701b96ee21632e8af6dcd1b393889dc4b151868d30`.
- The official repository describes the release as updated, deduplicated, and
  image-fixed. It is therefore `revised`, not `corrected`: issue 15 reports an
  isolated answer correction, issue 20 documents legacy scorer argument order,
  and issue 21 documents tuple/interval matching limitations. No accessible
  comprehensive corrected full release was found.
- DDSR retains theorem proofs but excludes them from automatic accuracy because
  upstream evaluated proofs manually and publishes no deterministic proof scorer.

### HiPhO

- The current official repository has 403 subquestions across 14 exams and 131
  PNG files. The published ICML benchmark and leaderboard use 360 subquestions
  across 13 exams; CPhO 2025 was added later and is an explicit extension.
- The pinned current source tree SHA-256 is
  `47f0041d39e57d9e7ec28669b21cd0cdc0218549719a59a90509a0c0e4ceeec8`.
  GitHub and Hugging Face carry semantically matching records with different
  JSON serialization, so DDSR materializes the pinned official GitHub commit.
- No accessible corrected full release or item-level correction ledger was
  found. DDSR records the source as `revised`: the official current release is
  authoritative, while published and current cohorts remain separately named.
- Six released subquestions have empty final-answer strings and therefore can
  receive only marking-scheme credit. The later `CPhO_2025_7_6` row has a
  malformed coarse reference in the maintained scorer; DDSR records that
  branch error and preserves its independent marking-scheme score.
- The official data repository does not ship an evaluator. DDSR pins the
  maintained VLMEvalKit implementation used by the reproducible Axion wrapper,
  including the English/Chinese prompts and answer-plus-step scoring policy.

### GPQA

- The pinned official password-protected archive contains 448 Main, 198
  Diamond, and 546 Extended question rows, plus 60 expert-metadata rows. DDSR
  validates all four files and uses Main by default.
- Its source tree SHA-256 is
  `77759a7dcb1909ba4f9962bdeb5a12be078c43070620464755b8fa19daeb0468`.
  The source includes pre-revision question and answer fields for provenance,
  while evaluation uses the current official fields as upstream does.
- The repository invites item corrections, and public issues identify candidate
  answer-key and baseline defects. No merged, accessible, comprehensively
  corrected full edition was found, so DDSR records this source as `original`
  rather than asserting that unresolved reports are verified corrections.

### MMLU

- The selected official release contains 14,042 test questions and 285 public
  development examples across 57 subjects. DDSR evaluates the complete suite.
- The official archive SHA-256 is
  `bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b`;
  its one-file materialized-tree SHA-256 is
  `e0d03a5240afbb89afdb74aa7b1017610d93d393ed26b8ea42906754c349f351`.
- MMLU-Redux and other audits repair or relabel subsets of MMLU, but do not
  provide an authoritative drop-in corrected edition of all 14,042 official
  test rows. They remain named derivatives rather than silently replacing the
  full benchmark.
- The original code uses next-token likelihoods from a legacy completion API.
  DDSR's provider-portable chat path preserves the prompt and records its
  generated-letter protocol separately; score comparisons must respect that
  distinction.

## Search boundary

This catalog was checked on 2026-11-20 against the official repositories and
dataset histories, public issues and forks, Hugging Face name searches, linked
papers, LOCA, and the Alliance audit release. An absence statement means no
accessible correction was found within that boundary; it is not a claim that
no private or future correction exists.
