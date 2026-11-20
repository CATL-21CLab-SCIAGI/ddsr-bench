# FrontierScience

FrontierScience is OpenAI's expert-level scientific-reasoning benchmark. Its
public gold set contains 100 Olympiad problems with short reference answers and
60 Research problems with ten-point rubrics, spanning physics, chemistry, and
biology. DDSR keeps the two official scoring contracts distinct.

## Data and preparation

```bash
python -m pip install -e '.[frontierscience]'
ddsr-bench action=fetch benchmark=frontierscience
ddsr-bench action=prepare benchmark=frontierscience \
  paths.input=datasets/frontierscience/gold \
  paths.output=tasks/frontierscience/gold
```

The public prompt enters the model request verbatim. Short answers and Research
rubrics remain verifier-side and are excluded from generation logs and teacher
trajectories. Prepared tasks are a validated offline fallback; no executable
verifier or Harbor container is required.

## Evaluation

```bash
ddsr-solve --config configs/jobs/frontierscience/vllm.yaml
ddsr-solve --config configs/jobs/frontierscience/vllm.yaml \
  --include-task-name TASK_GROUP_ID
```

Olympiad answers receive binary equivalence judgment. Research answers receive
0--10 rubric points, normalized to 0--1; seven points is the published success
threshold. The paper used GPT-5 with high reasoning effort and averaged 20
Olympiad or 30 Research trials per problem. The bundled one-attempt profile is
a smoke-friendly default; reproduce each track separately with its published
attempt count for comparable results.

The `tracks` and `subjects` lists under `agents[0].kwargs` are optional filters.
Changing the judge is a provider substitution and must be reported with scores.
See the [pipeline](PIPELINE.md), [dataset catalog](../../../DATASETS.md), and
[upstream record](../../../UPSTREAM.md) for exact revisions and limitations.

## Results and training data

Collection reports normalized score, threshold accuracy, Wald interval, and
breakdowns by track, subject, and attempt. The shared `native` and `full` SFT
views export only the model-under-test turn; judge messages and references never
become teacher samples.
