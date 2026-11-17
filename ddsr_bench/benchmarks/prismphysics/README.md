# PRISM-Physics

PRISM-Physics evaluates competition-level physics reasoning as a directed
acyclic graph of key formulas. A rule-based equation matcher identifies
reference formulas present in the model response; ancestor closure then gives
credit to every prerequisite of a matched formula. The result is a continuous
process score and a thresholded problem-level decision.

The adapter pins the official seven-shard dataset, text-only prompt, formula
engine, score propagation, and `0.8` correctness threshold. No accessible full
correction was found; the available independent audit covers only a frozen
100-problem subset, as recorded in the
[dataset catalog](../../../DATASETS.md). See the [PRISM pipeline](PIPELINE.md)
for the artifact flow. Shared client setup,
collection, and trajectory export are documented by the root
[workflow](../../../README.md).

## Data and prompt

Install the optional benchmark dependencies:

```bash
python -m pip install -e '.[prismphysics]'
```

The pinned Hugging Face release contains 1,401 records. Raw numeric IDs repeat
between the seven domain shards, so DDSR task names use `shard-id`, such as
`02-3006`. Public model inputs contain the rewritten context, subquestions, and
official formatting prompt. Reference solutions are discarded after loading;
only the formula DAG is retained verifier-side.

The exact upstream text prompt is preserved, including its leading/trailing
whitespace. It also preserves two observable upstream behaviors: the worked
example contains a form-feed created by a non-raw `\frac`, and released plural
`final_answer_instructions` are not appended because the upstream renderer
looks for singular keys.

The upstream driver defaults to text mode and separately supports multimodal
evaluation when a local image tree is supplied. The pinned Hugging Face revision
contains only the seven JSON shards and its README; it bundles no task JPEGs.
The JSON is a partial public image source: 90 image entries are absolute CDN
URLs, while 596 entries are unresolved root-relative paths and one entry is
malformed HTML. A live CDN JPEG was verified on 2026-11-17, but the complete
multimodal corpus could not be located. DDSR therefore implements the upstream
default text mode, retains image metadata only in the raw loader projection, and
omits unused image metadata from prepared instructions and generation records.
Four released question texts also contain relative Markdown image links inside
their context or subquestions. Those strings remain for byte-level prompt parity;
their corresponding relative assets were not found in the checked public trees.

Materialize and validate all seven pinned shards with:

```bash
ddsr-bench action=fetch benchmark=prismphysics
```

## Evaluation

### Static

Static evaluation is the normal path because the generated artifact is Markdown
text and no model-produced code executes:

```bash
ddsr-solve --config configs/jobs/prismphysics/vllm.yaml
```

Run one problem directly from the pinned dataset:

```bash
ddsr-solve \
  --config configs/jobs/prismphysics/aliyun.yaml \
  --include-task-name 02-3006
```

Equivalent `openai.yaml`, `bedrock.yaml`, and `aliyun.yaml` jobs read
`OPENAI_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, and `ALIYUN_API_KEY`. Static
outputs are written beneath `outputs/static/<job_name>/`. Resume with
`--resume` while keeping task selection and generation settings unchanged.

The scorer extracts relational `$$...$$` blocks, performs precision-aware
symbolic/unit equivalence matching, and applies ancestor closure. `reward` is
the fraction of credited DAG nodes. Status is `passed` when reward is at least
`0.8`, `different` for a successfully scored lower result, and `error` or
`timeout` when evaluation fails. Summaries report mean step score, thresholded
accuracy, and parse rate overall and by domain, shard, and attempt.

### Harbor (optional)

Harbor is unnecessary for scorer safety, but it is available when normalized
task directories, no-network verifier containers, or a common scheduler are
useful:

```bash
docker build \
  -f docker/prismphysics/Dockerfile \
  -t ddsr-bench-prismphysics:latest .

ddsr-bench action=prepare benchmark=prismphysics \
  harbor.image=ddsr-bench-prismphysics:latest \
  paths.input=datasets/prismphysics/official \
  paths.output=tasks/prismphysics/official

harbor run --config configs/jobs/prismphysics/vllm.yaml
```

Preparation reads all seven pinned shards from the supplied directory; omit
`paths.input` only to let the loader fetch them directly. Append
`--path tasks/prismphysics/official --include-task-name 02-3006` to run one
prepared task. Static evaluation falls back to those tasks when the source is
unavailable.

## Results and teacher data

Use the shared [collection](../../../README.md#-collect-results) and
[teacher-data export](../../../README.md#-export-teacher-data) commands with
either output mode. `native` and `full` SFT views preserve the single visible
answer call. Provider reasoning remains a separate trajectory field; the
reference formula DAG never enters response logs or exported samples.

Source revisions, the formula-engine license, and parser adaptation are recorded
in [UPSTREAM.md](../../../UPSTREAM.md). The evaluator inherits PRISM's documented
limitations for text-only answers, context-dependent equivalence, and long-tail
LaTeX parsing.
