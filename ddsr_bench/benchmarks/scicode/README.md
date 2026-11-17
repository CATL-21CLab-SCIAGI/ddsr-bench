# SciCode

SciCode asks a model to implement the ordered function-writing steps of one
scientific problem. Each generated step may depend on functions produced for
earlier steps, and the problem passes only when every evaluated step passes.

The local adapter follows SciCode's pinned prompt, dataset, target decoding, and
test behavior without requiring its Python package or a repository checkout.
See the [SciCode pipeline](PIPELINE.md) for the sequential generation and
verification flow.

Use the root [workflow](../../../README.md) for shared model setup,
collection, and export. Client configuration is covered by the
[generation guide](../../generation/README.md).

## Data and preparation

Install the optional data dependencies:

```bash
python -m pip install -e '.[scicode]'
```

Problem records load from the pinned SciCode-Verified v2.2 release. It contains
64 corrected, human-verified problems and 287 scored steps after excluding one
underdetermined original problem. The [dataset catalog](../../../DATASETS.md)
records the audit evidence and checksums. Download its matching HDF5 targets;
mixing targets from an earlier release changes grading behavior:

```bash
ddsr-bench action=fetch benchmark=scicode
```

Build the verifier image. It embeds the HDF5 targets so prepared task
directories contain only small public instructions and assertion files:

```bash
docker build -f docker/scicode/Dockerfile -t ddsr-bench-scicode:latest .

ddsr-bench \
  action=prepare \
  benchmark=scicode \
  paths.input=datasets/scicode/v2.2/data/problems_test.jsonl \
  paths.output=tasks/scicode/v2.2 \
  harbor.image=ddsr-bench-scicode:latest \
  harbor.timeout_sec=1800
```

## Evaluation

### Harbor

Run sequential generation and isolated corrected-test execution:

```bash
harbor run --config configs/jobs/scicode/vllm.yaml
```

Other client configurations live beside `vllm.yaml`. The default vLLM job
writes `outputs/harbor/scicode-verified`.

To evaluate one prepared problem, append
`--path tasks/scicode/v2.2 --include-task-name 5`. There is no
raw-data `ddsr-solve` shortcut: without the prepared assertions, HDF5 targets,
and container execution it could generate functions but could not evaluate them
under SciCode's contract.

### Recovery

Use Harbor's existing recovery command, not `ddsr-solve --resume`:

```bash
harbor job resume --job-path outputs/harbor/scicode-verified
```

Harbor 0.22.0 reloads the saved `config.json` and retains trials with valid saved
results, including exceptions. It deletes trial directories without `result.json`
and schedules replacements. **Back up the job outside its directory first** if
you need interrupted logs. To rerun a recorded cancellation, add
`--filter-error-type CancelledError`; this explicitly deletes matching trial
directories. Inspect recorded exception types before choosing that filter.

SciCode has no per-step generation checkpoints: restarting an unfinished trial
regenerates its steps from the beginning. Recovery does not require collection.
Automated tests cover Harbor's job reconciliation and an interrupted SciCode
agent. A live Docker smoke check also exercised two attempts of fixture problem
19 using a mock Chat Completions endpoint: interruption on call three followed
by resume produced five calls total, with the completed trial unchanged. Both
trials executed the verifier; deliberately incorrect mock code earned zero.
This checks recovery and container execution, not model accuracy.

### Static

Static evaluation is not supported because SciCode's score requires executing
the generated functions against its assertions and numeric targets.

## Results and teacher data

Use the shared [collection](../../../README.md#-collect-results) and
[teacher-data export](../../../README.md#-export-teacher-data) commands with the
job directory above. For SciCode, `native` and `full` both emit one training
sample per model-generated step. Bundled compatibility steps remain context and
are not training targets.

## Compatibility and limitations

SciCode recommends Inspect AI, and its implementation remains the compatibility
reference. ddsr-bench does not execute generated code through Inspect AI on the
host; Harbor keeps model access on the host and runs the corrected assertions in
an isolated no-network container.
