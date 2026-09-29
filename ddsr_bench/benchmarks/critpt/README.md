# CritPt

CritPt asks a model to derive a physics result and return it through a supplied
Python `answer(...)` template. A challenge contains one main problem and may
contain indexed subproblems.

ddsr-bench preserves CritPt's pinned official prompts and both generation
strategies: one-step derives and formats in one call; two-step derives first and
then applies the official formatting prompt. See the
[CritPt pipeline](PIPELINE.md) for the exact execution and verification paths.

Use the root [workflow](../../../README.md) for shared model setup,
collection, and export. Configure model access with the shared
[client guide](../../generation/README.md); CritPt generation is described below.

## Data and preparation

### Official problems

Download the pinned
[official challenge JSON files](https://github.com/CritPt-Benchmark/CritPt/tree/17c2545c302762d2f2d644d923ea4c301605cb08/data/public_test_challenges/json).
Provide the downloaded directory as `paths.input`:

```bash
ddsr-bench \
  action=prepare \
  paths.input=path/to/CritPt/data/public_test_challenges/json \
  paths.output=tasks/critpt-official
```

Preparation emits one task per main or indexed subproblem. The official public
set produces 70 main tasks. Public fields go into the model instruction;
reference code and optional testcases remain verifier-only.

### AI-generated problems

Supply one `problem.json` or a directory containing those files:

```bash
ddsr-bench action=prepare benchmark.source=critpt-ai \
  paths.input=/path/to/corpus paths.output=tasks/critpt-ai
```

The corpus's `auto` and `freestyle` labels mean answers graded by rules or an
LLM, respectively. Both become one task per record. Public statements, answer
instructions, and any code template go into `instruction.md`; references and
grading data stay in `tests/` and never enter model messages.

Preparation preserves the original ID and source path in `tests/reference.json`.
Paths are relative to the corpus's `CORPUS_INDEX.json` directory, or to the input
directory when that marker is absent. For rule grading, `reference_check` is a
light static check (`matched`, `different`, or `unresolved`), not a correctness
grade; all three outcomes allow preparation.

For teacher collection, copy a CritPt job, set `datasets[].path` to the prepared
AI tasks, choose a new `job_name`, and set `agents[].kwargs.style: one-step`.
Problem-specific answer instructions override defaults in
[ai.yaml](../../../configs/prompts/critpt/ai.yaml).
AI tasks currently support static generation and export only; customized grading,
two-step generation, and Harbor execution are not supported. Regenerate older
experimental tasks before use.

## Evaluation

### Harbor

For official problems, build the verifier image once (rebuild after code changes),
then run isolated generation and verification:

```bash
docker build -f docker/critpt/Dockerfile -t ddsr-bench-critpt:latest .
harbor run --config configs/jobs/critpt/vllm.yaml
```

Other client configurations live beside `vllm.yaml`. The default job directory
is `outputs/harbor/critpt-official`. To select one prepared problem, append
`--path tasks/critpt-official --include-task-name Challenge_1_main`.

### Static

For answer generation without Docker or execution of generated code:

```bash
ddsr-solve --config configs/jobs/critpt/vllm.yaml
```

The default job directory is `outputs/static/critpt-official`. Static evaluation
checks code structure where applicable, not answer correctness.

For a one-task check before running a batch, copy a job into ignored
`configs/local/`, choose a new `job_name`, and set one attempt and concurrency one:

```bash
mkdir -p configs/local
cp configs/jobs/critpt/aliyun.yaml configs/local/critpt-check.yaml
# Edit the copy's job_name, dataset path, and model settings first.
ddsr-solve --config configs/local/critpt-check.yaml \
  --include-task-name Challenge_1_main
```

With `job_name: critpt-check`, inspect `agent/response.json`,
`artifacts/answer.py` (if valid), and `validation/result.json` under
`outputs/static/critpt-check/Challenge_1_main__attempt-0/`.
Use another job name for a full batch and omit the task filter. Export credentials
in your shell; `.env` files are not loaded automatically.

### Long-context generation and recovery

Use `ddsr-solve --config` with `configs/jobs/critpt/vllm-long-context.yaml` or
`configs/jobs/critpt/aliyun-deepseek.yaml`. Both default to one attempt and
concurrency four; check token budgets against your endpoint's limits.

- **Budgets:** `sampling.max_tokens` sets the first call's output budget;
  `formatting_max_tokens` overrides the second call in two-step mode. Only vLLM
  uses `/tokenize` to fit output within `context_window - input - context_safety_tokens`.
  Set `require_complete_stages: true` to reject empty or non-stop completions.
- **Seeds:** `seed_base: 42` derives a stable seed per problem-attempt, shared
  by both calls (static only). CritPt sends it with either Aliyun request profile;
  provider determinism is not guaranteed. For Harbor, use `sampling.seed` instead.
- **Recovery:** `ddsr-solve --config JOB.yaml --resume` reuses completed trials
  (including failures) and `agent/stage-N.json` checkpoints. Keep settings and task
  selection unchanged except for concurrency. Partial `stage-N.stream.jsonl`
  journals are diagnostic, not completed answers.

<details>
<summary>Troubleshooting: resuming an older or relocated CritPt job</summary>

Use `resume_migration` only when an older run's output or dataset paths moved:

```yaml
resume_migration:
  path_prefixes:
    /old/critpt-eval/outputs: /shared/critpt/runs/outputs
    /old/tasks: /shared/critpt/tasks
```

Compatibility also covers former CritPt agent names and the transition from
`client_name: openai` to `aliyun` with `request_profile: openai`. It does not allow
changes to model, endpoint, prompts, seeds, or budgets. Original `job-config.yaml`
is preserved; mappings are recorded separately. Omit this setting for normal runs.

</details>

### Solve one challenge

CritPt can read one existing challenge JSON and generate its answer without
preparing a Harbor task:

```bash
ddsr-solve path/to/challenge.json outputs/example \
  --client CLIENT \
  --base-url BASE_URL \
  --api-key-env KEY_VARIABLE \
  --model MODEL \
  --style two-step \
  --reasoning-effort low \
  --max-tokens 32768 \
  --stream
```

The main problem is selected by default; pass `--include-task-name ID` for a
subproblem. This shortcut extracts and statically validates code but does not
execute `answer`, `real_answer`, or testcases, so its reward is null. The output
directory must not already exist.

## Results and teacher data

Use the shared [collection](../../../README.md#-collect-results) and
[teacher-data export](../../../README.md#-export-teacher-data) commands with
either job directory. `agent/response.json` retains the conversation and available
provider reasoning; artifacts use `answer.py` for code or `answer.txt` for the full
prose response. Static exports are unverified.

For official two-step trajectories, the `full` SFT view preserves both calls and
adds a derived one-step answer sample. AI one-step `native`, `full`, and `answer`
views preserve the same recorded prompt and answer.

## Submission

`benchmark.submission.backend` selects where saved answers are graded:

- `official` (default): submit saved answers to Artificial Analysis's grading API.
- `internal`: grade saved answers against private consensus references using
  Harbor-managed Docker workers; this produces internal match scores, not official accuracy.

Finish generation first. Both backends collect results if `summary.json` is absent
and reuse it otherwise. To refresh it, run `ddsr-bench action=collect paths.input=JOB_DIR`.
Submission makes no model calls and never overwrites an existing submission report.

Set `benchmark.submission.attempts=0` for attempt 0, or
`'benchmark.submission.attempts=[0,1,2,3,4]'` for five attempts.

### Official submission (Artificial Analysis)

Each selected attempt must contain exactly one answer for each of the 70 official
main problems. Set the API key and choose the official backend explicitly:

```bash
export ARTIFICIAL_ANALYSIS_API_KEY='...'
ddsr-bench action=submit benchmark=critpt \
  paths.input=JOB_DIR 'benchmark.submission.attempts=[0]' \
  benchmark.submission.backend=official
```

Writes `submission-0.json`. Selecting five attempts sends all 350 answers in
**one AA request** and saves `submission-0-1-2-3-4.json` with aggregate scores.
If a request fails, reconcile its outcome with AA before retrying or removing
`submission-ATTEMPTS.pending.json`: AA may already have accepted it.

### Internal submission

Requires Docker, the [CritPt image](#harbor), and a private reference file
(not distributed with this project):

```bash
ddsr-bench action=submit benchmark=critpt \
  paths.input=JOB_DIR 'benchmark.submission.attempts=[0]' \
  benchmark.submission.backend=internal \
  benchmark.submission.internal.references=/absolute/path/to/consensus-61-v2.json
```

Writes `submission-internal-0.json`, or `submission-internal-0-1-2-3-4.json` for
five selected attempts, after all grading finishes. Reports include per-attempt
and aggregate scores; missing or failed answers count as zero. These scores do
not change trial rewards or training filters. See the [scoring rules](evaluation/consensus/SCORING.md).

Grading defaults to four concurrent problems; adjust `benchmark.submission.internal.jobs`
for your resources. Worker limits and other settings are documented in the
[benchmark config](../../../configs/benchmark/critpt.yaml).

#### Reference caching

Successful reference outputs are cached by default in `outputs/cache/critpt-consensus`.
A new directory starts **cold**; reusing it provides a **warm** cache when settings
match. Answers always execute afresh. Override `benchmark.submission.internal.cache_dir`
to choose a private directory outside the job, or set it to `null` to disable disk caching.
In-memory reuse within a submission remains enabled.

See the consensus guide for [cache details](evaluation/consensus/README.md#reference-caching)
and maintainer-only [reference tools](evaluation/consensus/README.md#reference-maintenance).

## Compatibility and limitations

The pinned CritPt prompts and one-step and two-step conversations are checked
against the upstream renderer. Static evaluation cannot establish correctness;
use Harbor when a local reference or verifier testcase is available.
