# CMPhysBench

CMPhysBench evaluates graduate-level condensed-matter reasoning. A model writes
a derivation and places its final LaTeX expression in `\boxed{}`. The SEED metric
compares that expression with the official reference and awards partial credit.

The adapter pins both the upstream implementation and Hugging Face dataset. See
the [CMPhysBench pipeline](PIPELINE.md) for the data boundary and scoring flow.
Shared client setup, collection, and trajectory export are documented by the
root [workflow](../../../README.md#workflow). Client configuration is covered
by the [generation guide](../../generation/README.md).

## Data and preparation

Install the optional benchmark dependencies:

```bash
python -m pip install -e '.[cmphysbench]'
```

The pinned dataset contains 520 records. Only 100 currently have a non-empty
`final_answer`, and the official evaluator excludes records without one. By
default, ddsr-bench therefore loads all records but evaluates only those 100 and
leaves the other 420 unscored. It never derives references from the worked
solution text.

Public model input contains the question and relevant symbols. The worked
solution and `final_answer` remain evaluator-side and are excluded from model
messages, response logs, and training exports.

Preparing Harbor tasks is optional; the static path below reads the pinned
Hugging Face dataset directly.

## Evaluation

### Static

This is the normal CMPhysBench path. It loads the pinned Hugging Face dataset
directly and does not require task preparation, Harbor, or Docker. Generated
text is not executable code, so a timeout-controlled local worker can safely
extract the first balanced `\boxed{}` expression and apply SEED.

Results retain the official score from 0 to 100 and expose `score / 100` as the
framework reward. Summaries report mean SEED and exact accuracy overall, by
topic, by answer type, and by attempt.

Run one of the job configurations directly:

```bash
ddsr-solve --config configs/jobs/cmphysbench/vllm.yaml
```

Select one problem without downloading or preparing tasks separately:

```bash
ddsr-solve --config configs/jobs/cmphysbench/aliyun.yaml --include-task-name 1
```

Equivalent `openai.yaml`, `bedrock.yaml`, and `aliyun.yaml` jobs read
`OPENAI_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, and `ALIYUN_API_KEY`, respectively.
Without `--include-task-name`, each job evaluates all 100 gradeable records.
Static results are written beneath `outputs/static/<job_name>/`.

### Harbor (optional)

Harbor is unnecessary for isolation because CMPhysBench never executes model
output. Use it only when its standardized task format, scheduling, and container
records are useful alongside other benchmarks:

```bash
docker build -f docker/cmphysbench/Dockerfile \
  -t ddsr-bench-cmphysbench:latest .

ddsr-bench action=prepare benchmark=cmphysbench \
  harbor.image=ddsr-bench-cmphysbench:latest \
  paths.output=tasks/cmphysbench

harbor run --config configs/jobs/cmphysbench/vllm.yaml
```

Append `--path tasks/cmphysbench --include-task-name 1` to run only problem 1.

Preparation downloads the same pinned Hugging Face split and writes only the
100 gradeable problems. Harbor stores public input in `instruction.md` and the
official answer separately under `tests/reference.json`. The same job files are
accepted by `harbor run` and `ddsr-solve`; direct static runs ignore the prepared
dataset path while Hugging Face is reachable. If access fails with an I/O or
connection error, `ddsr-solve` falls back to the prepared tasks.

## Results and teacher data

Use the shared [collection](../../../README.md#4-collect-results) and
[teacher-data export](../../../README.md#5-export-teacher-data) commands with
either job directory. CMPhysBench `native` and `full` exports both preserve its
single recorded answer call, including separate reasoning metadata when the
client provides it.

## Compatibility and limitations

The local SEED source is adapted to the project style rather than redesigned.
Formatting, typing, and packaging changes must preserve upstream behavior, and
each scoring change requires parity tests against the pinned implementation.
The copied source retains its Apache-2.0 attribution.
