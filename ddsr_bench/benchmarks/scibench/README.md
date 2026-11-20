# SciBench

SciBench evaluates open-ended college-level problems from chemistry, physics,
and mathematics textbooks. DDSR preserves the current official release's 580
text targets and 151 image targets. Its 112 text and 26 image solved examples
are source support material, not additional held-out targets.

## Data and preparation

```bash
python -m pip install -e '.[scibench]'
ddsr-bench action=fetch benchmark=scibench
ddsr-bench action=prepare benchmark=scibench \
  paths.input=datasets/scibench/official \
  paths.output=tasks/scibench/official
```

The loader validates the complete repository data tree and embeds target images
as data URLs. Numeric references and original units remain verifier-side.

## Evaluation

```bash
ddsr-solve --config configs/jobs/scibench/vllm.yaml
ddsr-solve --config configs/jobs/scibench/vllm.yaml \
  --include-task-name text/atkins/000
```

The default `zero-cot` strategy reproduces upstream's unusual two-request
protocol: the derivation is generated once, then embedded into a fresh user
request ending in `Therefore, the answer is`. `zero`, `few`, and `few-cot` are
also supported; few-shot modes are text-only and use the first three released
solved examples from the target textbook.

The official scorer extracts the last `\boxed{...}` value and applies 5%
relative tolerance. It also rescales answers when scientific notation is part
of the stated unit. Static evaluation is complete because no model code is
executed. The repository provides image data but no image-specific inference
driver; DDSR applies the same released text prompt and appends the question
image as a multimodal message, and records this as a framework choice rather
than an upstream-exact claim.

See the [pipeline](PIPELINE.md), [dataset catalog](../../../DATASETS.md), and
[upstream record](../../../UPSTREAM.md) for known data and scorer limitations.
