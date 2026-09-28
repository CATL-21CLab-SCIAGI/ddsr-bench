# CritPt

CritPt asks a model to derive a physics result and return it through a supplied
Python `answer(...)` template. A challenge contains one main problem and may
contain indexed subproblems.

ddsr-bench preserves CritPt's pinned official prompts and both generation
strategies: one-step derives and formats in one call; two-step derives first and
then applies the official formatting prompt. See the
[CritPt pipeline](PIPELINE.md) for the exact execution and verification paths.

Use the root [workflow](../../../README.md) for shared model setup,
collection, and export. Direct generation and client configuration are covered
by the [generation guide](../../generation/README.md).

## Data and preparation

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

For a local AI corpus, use `benchmark.source=critpt-ai`:

Prepared instructions use `source: critpt-official` or `source: critpt-ai`, matching
the `CritPtSource` enum. Older `source: critpt` values are accepted as an alias for
`critpt-official`; newly prepared tasks use the canonical value.

```bash
ddsr-bench action=prepare benchmark.source=critpt-ai \
  paths.input=/path/to/auto/problem.json paths.output=tasks/critpt-ai-sample
```

Supply one file or a directory containing `problem.json` files. The corpus calls
rule-based grading `auto` and LLM-based grading `freestyle`; these labels describe
verification, not answer format. Record fields determine how data is normalized.
Problems whose answers are graded by rules have public fields matching
official tasks, with the Python
`code_template` separated from the remaining `answer_instructions`: problem-specific
one-stage guidance about reasoning, precision, and answer formatting. Empty or
null means use the prompt-config default; two-stage
generation will need explicit adaptation of this guidance. Each AI
record is a main problem; private reasoning checkpoints are not subproblems.
The final Python block of `reference_solution` becomes `tests/reference.py`.
`tests/reference.json` stores extracted `code`, `final_answer["answer"]` as
`snippet` (preserving its string, number, or list type), and private grading fields.
Its private `reference_check` field records static agreement between reference code and
`final_answer`: `matched`, `different`, or `unresolved`. All three allow preparation;
the latter two flag records for later review. This checks simple literals or matching
expression syntax, not mathematical correctness. Records for answers graded by an
LLM omit the field.
The template stays in
`instruction.md`; AI tasks do not need a separate `template.py`.
Shared generation instructions for AI-generated problems live in
[ai.yaml](../../../configs/prompts/critpt/ai.yaml), adapted from Auto_CritPt_Grader.
Problem-specific instructions take precedence. For answers graded by an LLM,
preparation stores `answer_instructions: null` when the source instructions match
the configured default; otherwise it retains the source text unchanged.
Changing the default does not reject source records. `code_template` and reference `code` are empty
strings, and no Python files are created. The unchanged `reference_answer` becomes
`snippet`; answer items, reasoning, and evidence remain in `tests/reference.json`.

Hash-based task IDs avoid collisions. `tests/reference.json` preserves the original
`problem_id` and `source_path`, relative to the ancestor containing
`CORPUS_INDEX.json`. Without that marker, paths are relative to the supplied
input directory (or the parent of a single input file). Join this path to your
local corpus root to locate the original record; it never enters the model prompt.
For one-stage teacher collection, copy a CritPt job config, set `datasets.path`
to the prepared AI tasks, choose a new `job_name`, and set the agent's
`kwargs.style: one-step`. Run `ddsr-solve --config path/to/job.yaml`.
AI and official prompts share `system_prompt()` and YAML rendering.
`ProblemSpec.grader` defaults to `null` (unspecified) for official tasks; AI
generation requires an explicit `rule` or `llm` value.
`SOLVER_SYSTEM_PROMPT` in `ai.yaml` is keyed by `grader` (`rule` or `llm`):
how the generated answer will be graded, not how the prompt is evaluated. AI uses the
upstream solver prompt selected by the prepared task's `grader` field, which the
loader derives from `grading_plan` or `answer_items`, not template presence. Output instructions
belong in the user message, labeled
“Final answer instructions” or “Response instructions”; extracted templates are
reattached after the instructions. Records preserve
the actual messages and provider reasoning. Official and AI resume checks both
compare saved request messages with the current prompts. Regenerate older experimental AI tasks to use
the renamed `answer_instructions` field and explicit `grader`.
Code artifacts use `answer.py`; prose uses `answer.txt` containing the full response,
not an extracted final answer. `response.json` retains the full teacher conversation.
Shared collection and trajectory export support both artifact formats. AI's
one-step `native`, `full`, and `answer` SFT views preserve the same recorded prompt
and answer. Provider reasoning remains separate in canonical trajectories.
Static validation gives no correctness reward; exports are marked unverified.
AI grading and Harbor execution remain deferred; `tests/test.sh` explicitly fails.

Preparation and static evaluation do not require Docker. For Harbor evaluation
or internal submission, build the verifier image once (and rebuild after code changes):

```bash
docker build -f docker/critpt/Dockerfile -t ddsr-bench-critpt:latest .
```

## Evaluation

### Harbor

For isolated generation and verification:

```bash
harbor run --config configs/jobs/critpt/vllm.yaml
```

Other client configurations live beside `vllm.yaml`. The default job directory
is `outputs/harbor/critpt-official`. To select one prepared problem, append
`--path tasks/critpt-official --include-task-name Challenge_1_main`.

### Static

For generation with non-executing AST validation:

```bash
ddsr-solve --config configs/jobs/critpt/vllm.yaml
```

The default job directory is `outputs/static/critpt-official`. Static evaluation
checks answer structure and safety but does not execute generated code.

Long-context stage budgets, streaming checkpoints, stable per-attempt seeds,
and resume are described in the [generation guide](../../generation/README.md#critpt-long-context-generation).
Long-generation examples are in `configs/jobs/critpt/vllm-long-context.yaml`
and `configs/jobs/critpt/aliyun-deepseek.yaml`; their `seed_base` requires
`ddsr-solve`. Machine paths and credentials
belong in ignored `configs/local/` and `.env` files.

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
either default job directory. Static results have no execution reward. For a
two-step trajectory, the `full` view preserves both recorded calls and adds a
derived one-step answer sample.

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

Requires Docker, the prepared CritPt image, and a private reference file
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
