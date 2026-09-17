# ddsr-bench

A unified evaluation and data framework for data-driven scientific reasoning.

ddsr-bench provides shared model access, evaluation orchestration, result
collection, and training-data export while preserving each benchmark's own data,
prompt, and verification contract.

## 🚀 Installation

Use Python 3.12 or newer. From this project's root:

```bash
mamba create -n ddsr-bench python=3.12 -y
mamba activate ddsr-bench
python -m pip install -e '.[dev]'
```

Benchmark-specific dependencies are optional. The selected benchmark guide
identifies the required extra and any external data or container preparation.

## 🧭 Choose a benchmark

Start with the [benchmark catalog](BENCHMARKS.md), then follow that benchmark's
README and pipeline. The catalog records which benchmarks support static
evaluation, Harbor isolation, result collection, and trajectory export.

Shared model-client setup is documented in the
[generation guide](ddsr_bench/generation/README.md). Job configurations follow
`configs/jobs/BENCHMARK/CLIENT.yaml` for local vLLM and hosted providers.

## 🧠 Check model access

For a local vLLM server, start the configured model:

```bash
ddsr-vllm configs/serving/vllm/macos-qwen38.yaml
```

In another terminal, verify model discovery and one chat completion:

```bash
ddsr-smoke \
  --client vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model ddsr-local
```

The generation guide includes equivalent examples for OpenAI, Amazon Bedrock,
and Alibaba Cloud Model Studio.

## ⚗️ Evaluate

After following the selected benchmark's preparation instructions, run its job
with Harbor:

```bash
harbor run --config configs/jobs/BENCHMARK/CLIENT.yaml
```

Harbor schedules independent problem attempts and isolates executable
verification. Benchmarks that do not require execution can also use the static
runner:

```bash
ddsr-solve --config configs/jobs/BENCHMARK/CLIENT.yaml
```

Not every benchmark supports both paths. Concrete commands, task selection, and
verification behavior belong to the benchmark guide.

## 📊 Collect results

Summarize a completed job without rerunning generation:

```bash
ddsr-bench action=collect paths.input=JOB_DIR
```

This writes `summary.json` and `summary.csv`, groups complete batches by attempt,
and reports incomplete trials separately.

## 🎓 Export teacher data

Convert completed trials into canonical trajectories and SFT samples:

```bash
ddsr-bench \
  action=export \
  paths.input=JOB_DIR \
  paths.output=DATASET_DIR \
  training.view=full
```

Export makes no model calls. See the
[training-data guide](ddsr_bench/training/README.md) for the available views and
their benchmark-specific behavior.

## 📚 Learn more

- [Benchmarks](BENCHMARKS.md): supported datasets and capabilities
- [Framework pipeline](PIPELINE.md): configuration layers and artifact flow
- [Contributing](CONTRIBUTING.md): repository structure and development checks
- [Upstream sources](UPSTREAM.md): pinned repositories, datasets, and provenance

CritPt additionally supports 61-problem [independent consensus scoring](ddsr_bench/benchmarks/critpt/evaluation/consensus/README.md)
for saved multi-attempt answers, with Docker or Linux worker isolation. Its
per-attempt match results and aggregate means remain separate from mainline
reward and training-data selection. See the [migration notes](ddsr_bench/benchmarks/critpt/MIGRATION.md)
for long-context runs and historical artifact compatibility.
