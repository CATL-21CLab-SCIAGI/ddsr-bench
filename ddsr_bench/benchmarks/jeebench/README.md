# JEEBench

JEEBench contains 515 IIT JEE-Advanced mathematics, physics, and chemistry
problems from 2016–2023. DDSR preserves all four official response types:
single-choice, multiple-choice with partial credit, integer, and numeric.

## Data and preparation

```bash
python -m pip install -e '.[jeebench]'
ddsr-bench action=fetch benchmark=jeebench
ddsr-bench action=prepare benchmark=jeebench \
  paths.input=datasets/jeebench/official \
  paths.output=tasks/jeebench/official
```

The selected edition is the pinned official archive. No authoritative corrected
English release or item-level errata ledger was accessible, so its status is
`unverified` rather than corrected.

## Evaluation

```bash
ddsr-solve --config configs/jobs/jeebench/vllm.yaml
ddsr-solve --config configs/jobs/jeebench/vllm.yaml \
  --include-task-name jee-adv-2016-paper-1-1
```

The default `cot` mode reproduces upstream's zero-shot chain-of-thought prompt.
`normal` and `one-shot` are also supported through `prompt_mode`; one-shot uses
the released subject/type demonstration. Self-consistency remains expressible
as multiple DDSR attempts rather than a separate model-client implementation.

Official scoring gives full credit for an exact multiple-choice answer, 0.25
per selected correct option for a nonempty proper subset with no wrong option,
and zero otherwise. Single-choice answers require an exact match. Integer and
numeric answers use absolute tolerance 0.01. Upstream releases extracted model
answers but not the extraction script; DDSR therefore documents and tests its
own final-answer parser instead of claiming extractor parity.

See the [pipeline](PIPELINE.md), [dataset catalog](../../../DATASETS.md), and
[upstream record](../../../UPSTREAM.md).
