# critpt-eval

A reproducible CritPt harness for answer generation, Harbor evaluation,
deterministic verification, trajectory export, and explicit Artificial Analysis
submission.

It reproduces CritPt's pinned one-step and two-step prompts, supports local and
hosted model endpoints, and keeps private verifier data outside conversations.

See the [pipeline flowchart](PIPELINE.md) for execution details and the
[generation guide](critpt_eval/generation/README.md) for client-specific setup.

## Installation

After cloning this repository, enter its root and create a Python 3.12
environment:

```bash
mamba create -n critpt-eval python=3.12 -y
mamba activate critpt-eval
python -m pip install -e '.[dev]'
```

This installs the four commands `critpt-eval`, `critpt-solve`, `critpt-smoke`,
and `critpt-vllm`, together with the development tools.

## Configuration

| Layer | Configuration | Purpose |
| --- | --- | --- |
| Serving | `configs/vllm/*.yaml` | Local model process and context limits |
| Generation | `agents[].kwargs` in `configs/job/*.yaml` | Client, model, prompt strategy, and sampling |
| Evaluation | Remaining job fields | Tasks, attempts, concurrency, and outputs |

One job file combines the generation and evaluation settings for a batch.

Run the following commands from the repository root.

## 1. Start and check the model endpoint

For the bundled local vLLM configuration:

```bash
critpt-vllm configs/vllm/macos-qwen38.yaml
```

In another terminal:

```bash
critpt-smoke \
  --client vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model critpt-local
```

The smoke command checks model discovery and one chat request before a benchmark
run. See the [generation guide](critpt_eval/generation/README.md) for OpenAI,
Amazon Bedrock, and single-problem commands.

## 2. Build the verifier image

This step is required only for Harbor:

```bash
docker build -f docker/Dockerfile -t critpt-eval:0.1.0 .
```

The image contains the verifier dependencies; model requests remain on the host.

## 3. Prepare the official tasks

Download the
[official CritPt challenges](https://github.com/CritPt-Benchmark/CritPt/tree/17c2545c302762d2f2d644d923ea4c301605cb08/data/public_test_challenges/json),
then provide their local directory as `paths.input`:

```bash
critpt-eval \
  action=prepare \
  paths.input=path/to/CritPt/data/public_test_challenges/json \
  paths.output=tasks/official
```

Preparation creates the 70 task directories expected by the bundled jobs under
`tasks/official`. Direct runs do not require preparation; see
[Run one problem](critpt_eval/generation/README.md#run-one-problem) for an
example.

## 4. Generate and evaluate

Run isolated execution-based evaluation with Harbor:

```bash
harbor run --config configs/job/vllm.yaml
```

Or generate candidates with static code validation only:

```bash
critpt-solve --config configs/job/vllm.yaml
```

Both commands use the same tasks and generation settings. `critpt-solve` checks
code structure and safety without executing answers; Harbor adds isolated
execution and comparison when verifier data is available.

The default outputs are `outputs/harbor/critpt-official` and
`outputs/static/critpt-official`, respectively. Replace `vllm.yaml` with
`openai.yaml` or `bedrock.yaml` after configuring that client.

## 5. Collect results

```bash
critpt-eval action=collect paths.input=outputs/harbor/critpt-official
```

Collection writes `summary.json` and `summary.csv`, groups complete batches by
attempt, and reports incomplete trials separately. It also accepts a static job
directory, whose rewards remain null.

## 6. Export teacher trajectories

```bash
critpt-eval \
  action=export \
  paths.input=outputs/harbor/critpt-official \
  paths.output=datasets/critpt-teacher \
  training.view=full
```

This writes `trajectories.jsonl` and `sft.jsonl`. For a two-step trajectory,
`full` emits derivation, formatting, and derived one-step answer samples. Use
`training.view=derivation`, `formatting`, or `answer` for one projection.
Export makes no model calls and retains quality and provenance metadata.

## 7. Submit one attempt

Submission requires one answer for each of the 70 official main problems and
never happens automatically:

```bash
export ARTIFICIAL_ANALYSIS_API_KEY='...'
critpt-eval \
  action=submit \
  paths.input=outputs/harbor/critpt-official \
  submission.attempt=0
```

The command rejects missing, duplicate, or mixed-attempt answers before making
the API request. The server response is saved as `submission-0.json` inside the
job directory, and an existing submission file is never overwritten.

## Results and isolation

Each trial retains its conversation, generated `answer.py`, validation result,
usage, model settings, latency, and hashes. Separate trial directories prevent
concurrent attempts from overwriting one another. Harbor verifies without
network access, and reference data never appears in model messages or logs.
