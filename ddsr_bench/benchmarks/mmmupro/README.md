# MMMU-Pro

MMMU-Pro evaluates multidisciplinary multimodal reasoning in 30 subjects. DDSR
implements the complete corrected release and keeps its three official settings
distinct: Standard-10, Standard-4, and Vision. The headline score is the mean
of Standard-10 and Vision accuracy; Standard-4 is an optional diagnostic.

## Data and preparation

```bash
python -m pip install -e '.[mmmupro]'
ddsr-bench action=fetch benchmark=mmmupro
ddsr-bench action=prepare benchmark=mmmupro \
  paths.input=datasets/mmmupro/corrected \
  paths.output=tasks/mmmupro/corrected
```

The setting names describe how upstream constructed the alternatives, not a
fixed number of choices in every released record: individual questions contain
2–12 options. Standard records interleave their embedded images according to
`<image N>` placeholders. Vision records use one screenshot containing the
complete question and choices, followed by the official Vision instruction.
Answer letters and released explanations remain verifier-side.

## Evaluation

```bash
ddsr-solve --config configs/jobs/mmmupro/vllm.yaml
ddsr-solve --config configs/jobs/mmmupro/vllm.yaml \
  --include-task-name 'standard10/test_Physics_359'
```

The bundled jobs run the official Standard-10 and Vision settings in
chain-of-thought mode with a 4,096-token output cap. Set `mode: direct` or add
`standard4` under `settings` for the other official protocols. Evaluation is
static: the official parser extracts an answer letter and no generated code is
executed, so Harbor adds no isolation value.

Upstream randomly guesses when a response cannot be parsed. DDSR preserves the
seed-42 fallback policy but derives the guess from the problem ID so concurrent
and resumed trials remain reproducible. Summaries report fallback counts.

See the [pipeline](PIPELINE.md), [dataset catalog](../../../DATASETS.md), and
[upstream record](../../../UPSTREAM.md) for source and protocol details.
