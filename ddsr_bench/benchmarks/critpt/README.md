# CritPt

CritPt asks a model to derive a physics result and return it through a supplied
Python `answer(...)` template. A challenge contains one main problem and may
contain indexed subproblems.

ddsr-bench preserves CritPt's pinned official prompts and both generation
strategies: one-step derives and formats in one call; two-step derives first and
then applies the official formatting prompt. See the
[CritPt pipeline](PIPELINE.md) for the exact execution and verification paths.

Use the root [workflow](../../../README.md#workflow) for shared model setup,
collection, and export. Direct generation and client configuration are covered
by the [generation guide](../../generation/README.md).

## Data and preparation

Download the pinned
[official challenge JSON files](https://github.com/CritPt-Benchmark/CritPt/tree/17c2545c302762d2f2d644d923ea4c301605cb08/data/public_test_challenges/json).
Build the verifier image, then provide the downloaded directory as
`paths.input`:

```bash
docker build -f docker/critpt/Dockerfile -t ddsr-bench-critpt:latest .

ddsr-bench \
  action=prepare \
  paths.input=path/to/CritPt/data/public_test_challenges/json \
  paths.output=tasks/critpt-official
```

Preparation emits one task per main or indexed subproblem. The official public
set produces 70 main tasks. Public fields go into the model instruction;
reference code and optional testcases remain verifier-only.

## Run

For isolated generation and verification:

```bash
harbor run --config configs/jobs/critpt/vllm.yaml
```

For generation with non-executing AST validation:

```bash
ddsr-solve --config configs/jobs/critpt/vllm.yaml
```

The default job directories are `outputs/harbor/critpt-official` and
`outputs/static/critpt-official`. Other client configurations live beside
`vllm.yaml`. To select one prepared problem through Harbor, append
`--path tasks/critpt-official --include-task-name critpt/Challenge_1_main`.

### Generate directly from one challenge

CritPt additionally supports generation from a challenge JSON without preparing
a Harbor task:

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

The main problem is selected by default; pass `--problem-id ID` for a
subproblem. This shortcut extracts and statically validates code but does not
execute `answer`, `real_answer`, or testcases, so its reward is null. The output
directory must not already exist.

## Results and teacher data

Use the shared [collection](../../../README.md#4-collect-results) and
[teacher-data export](../../../README.md#5-export-teacher-data) commands with
either default job directory. Static results have no execution reward. For a
two-step trajectory, the `full` view preserves both recorded calls and adds a
derived one-step answer sample.

## Artificial Analysis submission

Submission is explicit and requires exactly one answer for each of the 70
official main problems:

```bash
export ARTIFICIAL_ANALYSIS_API_KEY='...'
ddsr-bench \
  action=submit \
  paths.input=outputs/harbor/critpt-official \
  submission.attempt=0
```

The command rejects missing, duplicate, and mixed-attempt batches before making
one request. It writes `submission-0.json` without overwriting an existing
response. The payload retains CritPt's official `problem_id`, `generated_code`,
`model`, `generation_config`, and `messages` fields.
