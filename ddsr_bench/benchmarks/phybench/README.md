# PHYBench

PHYBench evaluates advanced physics reasoning. A model produces a step-by-step
solution and places its final readable LaTeX formula in `\boxed{}`. The
Expression Edit Distance (EED) metric gives partial credit by comparing the
symbolic expression tree with the selected reference.

The adapter pins LOCA's independently corrected 100-problem edition together
with the official prompt and EED implementation. LOCA reports expert review of
all 100 public-reference problems, identification of 20 wrong answers, and
corrections where possible; this is not an official PHYBench revision. See the
[dataset catalog](../../../DATASETS.md) for the evidence and limitations, and
the [PHYBench pipeline](PIPELINE.md) for the data and scoring flow. Shared client
setup, collection, and trajectory export are documented by the root
[workflow](../../../README.md).

## Data and preparation

Install the optional benchmark dependencies:

```bash
python -m pip install -e '.[phybench]'
```

The selected edition preserves all 100 public IDs and is DDSR's default because
it is the strongest accessible full semantic correction found. Parser-only
repairs, a 59-problem verified subset, and an independent 87-problem audited
subset remain separately named alternatives in the
[dataset catalog](../../../DATASETS.md).

Only the ID, physics tag, and question enter model messages. Worked solutions
and corrected reference answers remain evaluator-side and are absent from
response logs and training exports. Preparing Harbor tasks is optional because
static runs read the pinned source file directly.

Materialize and validate that corrected source with:

```bash
ddsr-bench action=fetch benchmark=phybench
```

## Evaluation

### Static

Static evaluation is the normal path because model output is text rather than
executable code. A process-isolated worker extracts the final balanced
`\boxed{}` expression and applies EED:

```bash
ddsr-solve --config configs/jobs/phybench/vllm.yaml
```

Select one problem without preparing tasks:

```bash
ddsr-solve --config configs/jobs/phybench/aliyun.yaml --include-task-name 133
```

Equivalent `openai.yaml`, `bedrock.yaml`, and `aliyun.yaml` jobs read
`OPENAI_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, and `ALIYUN_API_KEY`. Static
results are written beneath `outputs/static/<job_name>/`. Summaries report mean
EED and exact accuracy overall, by physics tag, and by attempt.

Resume with `ddsr-solve --config JOB.yaml --resume`, keeping settings and task
selection unchanged except for concurrency. Completed trials, including failures,
are reused. Interrupted trials are preserved under `.interrupted/` and restarted;
this can repeat an unfinished model request. Older jobs without `resume.json`
cannot be resumed safely.

### Harbor (optional)

Use Harbor when standardized task directories, container records, or a common
multi-benchmark scheduler are useful:

```bash
docker build -f docker/phybench/Dockerfile \
  -t ddsr-bench-phybench:latest .

ddsr-bench action=prepare benchmark=phybench \
  harbor.image=ddsr-bench-phybench:latest \
  harbor.timeout_sec=1800 \
  paths.input=datasets/phybench/loca/PHYBENCH_100_fixed.jsonl \
  paths.output=tasks/phybench/loca

harbor run --config configs/jobs/phybench/vllm.yaml
```

Append `--path tasks/phybench/loca --include-task-name 133` to run only problem
133. Preparation reads the supplied pinned JSONL file, writes public input to
`instruction.md`, and keeps the reference answer in `tests/reference.json`.
Omit `paths.input` only to let the loader use the same remote release. Static
runs fall back to these prepared tasks if the source cannot be imported or
reached.

## Results and teacher data

Use the shared [collection](../../../README.md#-collect-results) and
[teacher-data export](../../../README.md#-export-teacher-data) commands with
either job directory. The `native` and `full` SFT views both preserve the
single visible solution call. Provider reasoning remains separate in canonical
trajectories and is not copied into visible SFT completions.

## Compatibility and limitations

The official repository publishes EED but no inference driver. The model prompt
is reproduced from Appendix D of the PHYBench paper and sent as one user
message. The paper specifies `temperature=0.6`, `top_p=0.95`, and 32,768 output
tokens for local inference, while its API evaluations use provider defaults.
Provider job files document any explicit project choices that differ.

The paper defines boxed-answer extraction but does not publish that driver; this
adapter selects the final balanced box before applying the official EED code.
EED also short-circuits identical strings before parsing. In a pinned-data
compatibility probe, all 100 corrected references self-score by that path, but
only 77 self-score after harmless trailing whitespace forces parsing. This is a
metric limitation, not evidence that LOCA corrected only 77 records.
Upstream provenance and pinned revisions are recorded in
[UPSTREAM.md](../../../UPSTREAM.md). The adapted EED source retains its MIT
attribution and is protected by parity tests.
