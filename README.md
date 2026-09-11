# ddsr-bench

A unified evaluation and data framework for data-driven scientific reasoning.

ddsr-bench separates reusable model access, evaluation orchestration, result
collection, and training-data export from each benchmark's data and verification
contract.

Start with the [benchmark catalog](BENCHMARKS.md), then use the selected
benchmark's guide for its data and run configuration. The
[framework pipeline](PIPELINE.md) explains how the shared and benchmark-specific
layers interact.

## Installation

After cloning this repository, enter its root and create a Python 3.12
environment:

```bash
mamba create -n ddsr-bench python=3.12 -y
mamba activate ddsr-bench
python -m pip install -e '.[dev]'
```

This installs the shared `ddsr-bench`, `ddsr-solve`, `ddsr-smoke`, and
`ddsr-vllm` commands. Benchmark guides identify optional dependencies and
whether evaluation uses Harbor or the static `ddsr-solve` runner.

## Repository layout

```text
configs/jobs/             benchmark and client job configurations
configs/serving/          local model-serving configurations
ddsr_bench/benchmarks/    benchmark adapters and documentation
ddsr_bench/generation/    shared model clients
ddsr_bench/training/      shared trajectory and SFT export
docker/                   benchmark verifier images
```

Each benchmark package follows the same capability boundary:

```text
data/          source loading and public/private projection
generation/    benchmark prompt and generation policy
evaluation/    task preparation, agent, and verifier
result.py      adapter for shared result collection
trajectory.py  adapter for shared training-data export
```

## Configuration

| Layer | Location | Controls |
| --- | --- | --- |
| Serving | `configs/serving/` | Local model process and context limits |
| Generation | `agents[].kwargs` in a job | Client, model, prompts, and sampling |
| Evaluation | Remaining job fields | Tasks, attempts, concurrency, and outputs |

Job configurations follow `configs/jobs/BENCHMARK/CLIENT.yaml`. Client setup is
documented in the [generation guide](ddsr_bench/generation/README.md).
Each job names its benchmark; the explicit benchmark registry selects the
supported runner, while `configs/benchmark/` remains the single source for
dataset and benchmark defaults.

## Workflow

Run these commands from the repository root.

### 1. Prepare

Choose a benchmark from [BENCHMARKS.md](BENCHMARKS.md) and follow its guide to
perform any required data, dependency, verifier-image, and task preparation.
These steps intentionally remain benchmark-specific.

### 2. Check the model endpoint

For a local vLLM server, start a serving configuration:

```bash
ddsr-vllm configs/serving/vllm/macos-qwen38.yaml
```

Check model discovery and one chat request in another terminal:

```bash
ddsr-smoke \
  --client vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model ddsr-local
```

### 3. Generate and evaluate

Select one of the prepared job configurations:

```bash
harbor run --config configs/jobs/BENCHMARK/CLIENT.yaml
```

Harbor schedules independent problem-attempt trials and runs verifier code in
no-network containers. Some benchmarks expose an additional non-executing mode;
their guides document its command and limitations.

To run one prepared problem, append `--path tasks/TASK_SET` and
`--include-task-name TASK_NAME`. Harbor filters local datasets by task
directory name; benchmark guides provide concrete IDs.

### 4. Collect results

```bash
ddsr-bench action=collect paths.input=JOB_DIR
```

Collection writes `summary.json` and `summary.csv`, groups complete batches by
attempt, and reports incomplete trials separately.

### 5. Export teacher data

```bash
ddsr-bench \
  action=export \
  paths.input=JOB_DIR \
  paths.output=DATASET_DIR \
  training.view=full
```

Export writes `trajectories.jsonl` and `sft.jsonl` without making model calls.
`native` preserves recorded calls; `full` may add benchmark-specific derived
samples. One export must contain exactly one benchmark. See the
[training-data guide](ddsr_bench/training/README.md) for the available views.

## Trials and outputs

A trial is one complete evaluation of one problem for one attempt. Concurrency
changes how many trials run at once, not how attempts are grouped. Separate
trial directories prevent concurrent runs from overwriting one another.

Each trial retains its conversation, generated benchmark artifact, validation
result, usage, model settings, and latency. Trajectory export adds content and
source hashes. Private verifier data stays outside model messages and agent
logs. See the [framework pipeline](PIPELINE.md) for the full artifact flow.
