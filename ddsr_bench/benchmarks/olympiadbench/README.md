# OlympiadBench

OlympiadBench contains 8,476 bilingual mathematics and physics competition
problems: 6,728 open-ended problems and 1,748 theorem proofs. The pinned revised
release includes 4,869 multimodal problems with embedded images.

## Data and preparation

```bash
python -m pip install -e '.[olympiadbench]'
ddsr-bench action=fetch benchmark=olympiadbench
ddsr-bench action=prepare benchmark=olympiadbench \
  paths.input=datasets/olympiadbench/official \
  paths.output=tasks/olympiadbench/official
```

DDSR validates all 18 official subsets, their composition, image payloads, and
image placeholders. Public prompts and images are separated from final answers,
solutions, and scoring tolerances in prepared tasks.

## Evaluation

```bash
ddsr-solve --config configs/jobs/olympiadbench/vllm.yaml
ddsr-solve --config configs/jobs/olympiadbench/vllm.yaml \
  --include-task-name OE_TO_maths_en_COMP/0
```

The official deterministic scorer evaluates open-ended responses using boxed
answer extraction, numeric tolerance, symbolic equivalence, equations, intervals,
multiple answers, units, and percentage variants. The official release does not
define an automatic theorem-proof verifier: DDSR generates those responses and
reports them as `ungraded`, never as incorrect. Accuracy therefore uses only the
gradable open-ended population.

Optional `kinds`, `modalities`, `subjects`, `languages`, and `sources` lists under
`agents[0].kwargs` filter the full benchmark. `include_images: false` selects
text-only problems; it never removes an image from a multimodal problem.

## Provenance and limitations

Prompts follow the pinned upstream inference harness. The newer official scorer
is applied reference-first; this avoids the argument-direction bug in the legacy
judge script. DDSR uses `latex2sympy2-extended` as a maintained parser-compatible
replacement for SymPy's narrowly pinned ANTLR runtime while preserving the
published comparison order.

The dataset is marked `revised`, not `corrected`: upstream fixed and deduplicated
the original release, but public issues still identify isolated answer and scorer
defects. See the [pipeline](PIPELINE.md), [dataset catalog](../../../DATASETS.md),
and [upstream record](../../../UPSTREAM.md).

## Results and training data

Collection reports generated, graded, ungraded, passed, error, and accuracy
counts overall and by problem kind, modality, subject, language, source, and
attempt. The `native` and `full` trajectory views retain the exact text or
multimodal model turn without verifier-only solutions or answers.
