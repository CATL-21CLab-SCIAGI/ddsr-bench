# SciCode

SciCode is integrated as a separate benchmark because its contract differs from
CritPt: one main problem contains ordered coding steps, each step may depend on
previous generated functions, and the main problem passes only when every step
passes.

The adapter reads the upstream dataset but does not require its Python package or
a repository checkout. It locally follows the pinned prompt and evaluation
behavior while Harbor provides process isolation. CritPt prompts, `answer()`
validation, and reference-function comparison do not apply to SciCode.

## Data

Install the SciCode data dependencies declared by this project:

```bash
python -m pip install -e '.[scicode]'
```

The upstream Python package is not required. This adapter implements its pinned
prompt, dataset, target-decoding, and evaluation behavior locally.

SciCode loads its problem records from `SciCode1/SciCode` on Hugging Face. Its
numeric targets are stored separately in `test_data.h5`; download that file as
directed by the upstream repository and keep it outside version control.

Place the file at `datasets/scicode/test_data.h5`, then build the shared verifier
image once:

```bash
docker build -f docker/scicode/Dockerfile -t ddsr-bench-scicode:latest .
```

Each Harbor container reads `/opt/scicode/test_data.h5` from this image. The
prepared task directories contain only their small assertion files.

## Run

Prepare the validation split without a repository checkout:

```bash
ddsr-bench action=prepare benchmark=scicode \
  paths.output=tasks/scicode-validation \
  harbor.image=ddsr-bench-scicode:latest \
  harbor.timeout_sec=1800
```

Then run generation and isolated verification:

```bash
harbor run --config configs/jobs/scicode/vllm.yaml
```

For the tested Bedrock/Luna configuration:

```bash
export AWS_BEARER_TOKEN_BEDROCK='...'
harbor run --config configs/jobs/scicode/bedrock.yaml
```

Override `benchmark.split=test` and the task path in the job file for the test
split.

Collect completed trials with the same command used by other benchmarks:

```bash
ddsr-bench action=collect paths.input=outputs/harbor/scicode-validation-vllm
```

The job directory receives the shared `summary.json` and `summary.csv` files.

## Inspect AI

SciCode recommends Inspect AI and its implementation remains our compatibility
reference. We do not run its generated-code subprocess directly on the host.
The Harbor adapter instead keeps model access on the host and executes the
official tests in an isolated container.

Upstream: https://github.com/scicode-bench/SciCode
