# CMT-Benchmark

CMT-Benchmark evaluates formula-level condensed-matter reasoning across methods
such as DMRG, exact diagonalization, Hartree–Fock, PEPS, QMC, shell models, and
VMC. A model returns one or more expressions inside a final `\boxed{}` answer;
the deterministic scorer checks mathematical equivalence with the reference.

The adapter pins the maintained official v2.0.0 dataset, whose changelog records
expert-reviewed fixes to 10 problems, together with the official system
instruction and scoring behavior. See the [dataset catalog](../../../DATASETS.md)
for edition provenance and the [CMT pipeline](PIPELINE.md) for the data and
scoring flow. Shared client setup, collection, and trajectory export are
documented by the root [workflow](../../../README.md).

## Data and preparation

Install the optional benchmark dependencies:

```bash
python -m pip install -e '.[cmtbench]'
```

The pinned Hugging Face file contains 50 unique, gradeable problems. Public
model input contains only the problem type and raw prompt. Official solutions,
declared parameters, and function names remain evaluator-side and are excluded
from model messages, response logs, and training exports.

Preparing Harbor tasks is optional because static runs read the pinned dataset
directly.

For reproducible local preparation, materialize and validate the pinned source:

```bash
ddsr-bench action=fetch benchmark=cmtbench
```

## Evaluation

### Static

Static evaluation is the normal path because generated answers are text, not
executable code. A bounded worker extracts the final `\boxed{}` or `\fbox{}`
content and routes it through numeric, parameterized-symbolic, or functional
comparison:

```bash
ddsr-solve --config configs/jobs/cmtbench/vllm.yaml
```

Select one problem without preparing tasks:

```bash
ddsr-solve --config configs/jobs/cmtbench/aliyun.yaml --include-task-name 39
```

Equivalent `openai.yaml`, `bedrock.yaml`, and `aliyun.yaml` jobs read
`OPENAI_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, and `ALIYUN_API_KEY`. Static
results are written beneath `outputs/static/<job_name>/`. Summaries report the
fixed-denominator pass rate and parse rate overall, by problem type, and by
attempt. A parsed but inequivalent answer has status `different`; parser or
evaluation failures receive zero reward.

Resume with `ddsr-solve --config JOB.yaml --resume`, keeping settings and task
selection unchanged except for concurrency. Completed trials, including
failures, are reused. Interrupted trials are preserved under `.interrupted/`
and restarted, which can repeat an unfinished model request.

### Harbor (optional)

Use Harbor only when normalized task directories, container records, or a
common multi-benchmark scheduler are useful:

```bash
docker build -f docker/cmtbench/Dockerfile -t ddsr-bench-cmtbench:latest .

ddsr-bench action=prepare benchmark=cmtbench \
  harbor.image=ddsr-bench-cmtbench:latest \
  paths.input=datasets/cmtbench/v2/cmt_data.jsonl \
  paths.output=tasks/cmtbench/v2

harbor run --config configs/jobs/cmtbench/vllm.yaml
```

Append `--path tasks/cmtbench/v2 --include-task-name 39` to run only problem 39.
Preparation reads the supplied pinned JSONL file, stores public input in
`instruction.md`, and keeps private scoring fields under `tests/reference.json`.
Omit `paths.input` only to let the loader use the same remote release. Static
runs fall back to prepared tasks if the source cannot be imported or reached.

## Results and teacher data

Use the shared [collection](../../../README.md#-collect-results) and
[teacher-data export](../../../README.md#-export-teacher-data) commands with
either job directory. The `native` and `full` SFT views preserve the single
visible answer call. Provider reasoning stays separate in canonical trajectories
and is not copied into visible SFT completions.

## Compatibility and limitations

The exact official system instruction is followed by the dataset's raw prompt.
Upstream OpenAI-compatible calls use provider defaults; DDSR job files explicitly
select their documented output limits and sampling policy.

The scorer is reimplemented rather than copied because its pinned repository
does not include a standalone source license. Its extraction and equivalence
decisions are checked differentially against the pinned implementation, including
all 50 v2 references. Source revisions and the license boundary are recorded in
[UPSTREAM.md](../../../UPSTREAM.md).
