# PHYBench

PHYBench evaluates advanced physics reasoning. A model produces a step-by-step
solution and places its final readable LaTeX formula in `\boxed{}`. The
Expression Edit Distance (EED) metric gives partial credit by comparing the
symbolic expression tree with the official reference.

The adapter pins the official dataset, prompt, and EED implementation. See the
[PHYBench pipeline](PIPELINE.md) for the data and scoring flow. Shared client
setup, collection, and trajectory export are documented by the root
[workflow](../../../README.md).

## Data and preparation

Install the optional benchmark dependencies:

```bash
python -m pip install -e '.[phybench]'
```

The canonical Hugging Face file contains 500 unique problems. One hundred have
official solutions and answers and can be scored locally; the other 400 are
generation-only. Default evaluation selects the 100 gradeable records.

Only the ID, physics tag, and question enter model messages. Worked solutions
and reference answers remain evaluator-side and are absent from response logs
and training exports. Preparing Harbor tasks is optional because static runs
read the pinned Hugging Face file directly.

## Evaluation

### Static

Static evaluation is the normal path because model output is text rather than
executable code. A process-isolated worker extracts the final balanced
`\boxed{}` expression and applies EED:

```bash
ddsr-solve --config configs/jobs/phybench/vllm.yaml
```

Select one problem without preparing tasks:

```bash
ddsr-solve --config configs/jobs/phybench/aliyun.yaml --include-task-name 133
```

Equivalent `openai.yaml`, `bedrock.yaml`, and `aliyun.yaml` jobs read
`OPENAI_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, and `ALIYUN_API_KEY`. Static
results are written beneath `outputs/static/<job_name>/`. Summaries report mean
EED and exact accuracy overall, by physics tag, and by attempt.

### Harbor (optional)

Use Harbor when standardized task directories, container records, or a common
multi-benchmark scheduler are useful:

```bash
docker build -f docker/phybench/Dockerfile \
  -t ddsr-bench-phybench:latest .

ddsr-bench action=prepare benchmark=phybench \
  harbor.image=ddsr-bench-phybench:latest \
  harbor.timeout_sec=1800 \
  paths.output=tasks/phybench

harbor run --config configs/jobs/phybench/vllm.yaml
```

Append `--path tasks/phybench --include-task-name 133` to run only problem 133.
Preparation writes public input to `instruction.md` and the reference answer to
`tests/reference.json`. Static runs fall back to these prepared tasks if the
pinned dataset cannot be imported or reached.

## Results and teacher data

Use the shared [collection](../../../README.md#-collect-results) and
[teacher-data export](../../../README.md#-export-teacher-data) commands with
either job directory. The `native` and `full` SFT views both preserve the
single visible solution call. Provider reasoning remains separate in canonical
trajectories and is not copied into visible SFT completions.

## Compatibility and limitations

The official repository publishes EED but no inference driver. The model prompt
is reproduced from Appendix D of the PHYBench paper and sent as one user
message. The paper specifies `temperature=0.6`, `top_p=0.95`, and 32,768 output
tokens for local inference, while its API evaluations use provider defaults.
Provider job files document any explicit project choices that differ.

The paper defines boxed-answer extraction but does not publish that driver; this
adapter selects the final balanced box before applying the official EED code.
Upstream provenance and pinned revisions are recorded in
[UPSTREAM.md](../../../UPSTREAM.md). The adapted EED source retains its MIT
attribution and is protected by parity tests.
