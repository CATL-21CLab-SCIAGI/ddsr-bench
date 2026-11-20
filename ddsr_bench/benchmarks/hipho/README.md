# HiPhO

HiPhO evaluates worked solutions to recent high-school physics Olympiads with
official final answers and step-level marking schemes. The current official
release contains 403 subquestions from 14 exams and 131 PNG assets. The
published leaderboard cohort contains 360 subquestions from 13 exams; the later
CPhO 2025 addition is retained as a named extension rather than silently mixed
into published scores.

## Data and preparation

```bash
python -m pip install -e '.[hipho]'
ddsr-bench action=fetch benchmark=hipho
ddsr-bench action=prepare benchmark=hipho \
  paths.input=datasets/hipho/official \
  paths.output=tasks/hipho/official
```

DDSR validates every exam row, unique problem ID, source label, referenced PNG,
exam count, and published-cohort count. Prepared instructions contain only the
question, data sheet, public metadata, and images. Answers, points, and marking
criteria stay verifier-side.

## Evaluation

```bash
ddsr-solve --config configs/jobs/hipho/vllm.yaml
ddsr-solve --config configs/jobs/hipho/vllm.yaml \
  --include-task-name IPhO_2025_1_A_1
```

Static evaluation is the official path: generated code is never executed, so a
Harbor container adds no evaluation isolation. The maintained VLMEvalKit scorer
computes answer-level credit and Gemini-2.5-Flash step-level credit, then uses
the larger raw point score. The job configurations reproduce temperature `0.6`
and eight independent attempts; the output cap should be raised to the largest
limit supported by the evaluated model.

`cohort: published` reproduces the 13-exam leaderboard population. Set it to
`current` to include CPhO 2025. Optional `exams`, `fields`, `modalities`, and
`languages` lists filter either cohort. `include_images: false` selects only
rows without images; it never strips an image from a problem.

The maintained scorer predates CPhO. DDSR invokes its unchanged fine and coarse
branches separately: on the published cohort this is identical to upstream;
for a malformed CPhO coarse reference, the recorded coarse branch receives zero
while step-level scoring remains available instead of losing the whole trial.

## Scores and trajectories

The official leaderboard statistic is the arithmetic mean of the exams' raw
point totals, not overall percentage accuracy. Collection therefore reports
`leaderboard_points_mean` and the diagnostic `score_rate` separately, for each
attempt and for both the current and published cohorts.

The `native` and `full` training views preserve the exact image-first user turn
and model response. They exclude final answers, point allocations, marking
criteria, and judge prompts. HiPhO's source is evaluation data; users must make
their own contamination and training-license assessment before SFT use.

See the [pipeline](PIPELINE.md), [dataset catalog](../../../DATASETS.md), and
[upstream record](../../../UPSTREAM.md) for exact revisions and limitations.
