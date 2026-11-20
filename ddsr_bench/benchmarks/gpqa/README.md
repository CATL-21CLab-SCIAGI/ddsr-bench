# GPQA

GPQA evaluates graduate-level multiple-choice reasoning in biology, chemistry,
and physics. DDSR pins the official repository archive and preserves its three
question cohorts: Main (448), Diamond (198), and Extended (546). The 60-row
Experts CSV is contributor metadata, not a fourth question cohort.

## Data and preparation

```bash
python -m pip install -e '.[gpqa]'
ddsr-bench action=fetch benchmark=gpqa
ddsr-bench action=prepare benchmark=gpqa \
  paths.input=datasets/gpqa/official \
  paths.output=tasks/gpqa/official
```

DDSR reads the password-protected official archive without extracting it,
validates every cohort, and reproduces upstream's seeded answer-choice shuffle.
Prepared tasks contain the shuffled public question and choices. The answer and
expert explanation remain verifier-side.

## Evaluation

```bash
ddsr-solve --config configs/jobs/gpqa/vllm.yaml
ddsr-solve --config configs/jobs/gpqa/vllm.yaml \
  --include-task-name rec055vn3qEqKHHTc
```

The bundled jobs select the Main cohort. Change `cohort` to `diamond` or
`extended` when the raw archive is available. Optional `domains` and
`subdomains` lists filter the selected cohort.

Generation uses the official zero-shot system and user prompts. Scoring applies
the official sampled-answer parser and exact A-D accuracy. It never executes
model output, so static evaluation is the normal path and Harbor isolation is
unnecessary.

## Scores and trajectories

Collection reports fixed-denominator accuracy overall, by attempt, domain, and
subdomain; malformed answers receive zero. The `native` and `full` trajectory
views preserve the exact prompt and response without verifier-only explanations
or answer keys.

See the [pipeline](PIPELINE.md), [dataset catalog](../../../DATASETS.md), and
[upstream record](../../../UPSTREAM.md) for exact revisions and limitations.
