# HLE-Verified

Humanity's Last Exam (HLE) is a multidisciplinary benchmark of difficult,
closed-ended academic questions. HLE-Verified audits the 2,500-item release and
assigns each item to Gold, Revision, or Uncertain. DDSR selects the 1,811 Gold
and Revision items as its certified default; the 689 unresolved Uncertain items
remain in the pinned source but are not scored by default.

The default selection contains 1,372 exact-match and 439 multiple-choice
answers. It is multimodal: 211 selected questions include embedded image data.
See the [dataset catalog](../../../DATASETS.md) for edition provenance and the
[HLE pipeline](PIPELINE.md) for the evaluation flow.

## Data and preparation

Install the loader extra and fetch the pinned edition:

```bash
python -m pip install -e '.[hle]'
ddsr-bench action=fetch benchmark=hle
```

This materializes all three verified classes at `datasets/hle/verified/`.
Questions and released images are public model inputs. Correct answers and
rationales, including 272 selected rationale images, remain evaluator-side and
are excluded from generation logs and training exports.

Static evaluation reads the pinned Hugging Face dataset directly. Standardized
tasks provide a validated offline fallback for the selected classes:

```bash
ddsr-bench action=prepare benchmark=hle \
  paths.input=datasets/hle/verified \
  paths.output=tasks/hle/verified
```

The tasks contain no executable verifier. HLE scoring calls an LLM judge on the
host, so Harbor and Docker provide no execution-isolation benefit.

## Evaluation

Configure one model endpoint and one judge endpoint, then run the static job:

```bash
ddsr-solve --config configs/jobs/hle/vllm.yaml
```

Run one item by its original ID:

```bash
ddsr-solve --config configs/jobs/hle/vllm.yaml \
  --include-task-name 668825f80a642802bdfeadfa
```

The OpenAI profile pins the official HLE judge, `o3-mini-2025-01-31`.
The vLLM, Bedrock, and Aliyun profiles show provider substitutions; their scores
are operationally useful but not directly comparable to official-judge scores.
Set `include_images: false` only for a text-only endpoint; doing so changes the
evaluated population. API profiles consume quota.

The official HLE-Verified report averages five independent rollouts per item,
so bundled jobs use `n_attempts: 5`. Reduce it for smoke tests, but label the
result accordingly. Outputs are written under `outputs/static/<job_name>/` and
can be resumed with `--resume`.

## Metrics and teacher data

Collection reports fixed-denominator accuracy, the official Wald 95% half-width,
RMS calibration error, unjudged count, and breakdowns by verified class,
category, answer type, and attempt. Missing generation or judge output counts as
wrong.

Use the shared [collection](../../../README.md#-collect-results) and
[teacher-data export](../../../README.md#-export-teacher-data) commands. The
`native` and `full` views preserve the single HLE generation call, including
public image parts and separate provider reasoning.

## Compatibility and limitations

DDSR copies the official HLE response-format and judge prompts byte-for-byte,
including the legacy o1 system-role exception, and preserves the official
calibration implementation's skipped final bin. HLE's LLM judge is inherently
model-dependent; DDSR does not replace it with exact matching or claim parity
between different judges.

HLE-Verified's repository and dataset card contain no explicit dataset license
at the pinned revisions. DDSR downloads the data for local use and does not
redistribute it. The original HLE evaluation code and copied prompts are MIT;
the HLE-Verified paper is CC BY 4.0. Exact revisions, counts, hashes, and source
boundaries are recorded in [UPSTREAM.md](../../../UPSTREAM.md).
