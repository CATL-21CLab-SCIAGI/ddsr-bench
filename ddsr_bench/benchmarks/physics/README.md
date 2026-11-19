# PHYSICS

PHYSICS evaluates university-level physics problem solving across atomic
physics, classical mechanics, electromagnetism, optics, quantum mechanics, and
thermodynamics/statistical mechanics. A response contains an explanation,
ordered final answers, and one confidence value. An LLM judge independently
scores every reference answer.

DDSR selects the official `verified` edition, PHYSICS-Verified: 1,034 audited
PhD-qualifying-exam problems with 2,559 scored answers. It contains 240
validation and 794 test problems; 248 problems include embedded images. See the
[dataset catalog](../../../DATASETS.md) for edition provenance and the
[PHYSICS pipeline](PIPELINE.md) for the evaluation flow.

## Data and preparation

Install the optional loader dependency and fetch the pinned edition:

```bash
python -m pip install -e '.[physics]'
ddsr-bench action=fetch benchmark=physics
```

This materializes `datasets/physics/verified/`. Public requests contain the
question and any released image data. Solutions, final answers, labels, and
answer types remain evaluator-side and are excluded from generation logs and
training exports.

Standardized tasks are optional because static evaluation reads the pinned
dataset directly. They provide a validated offline fallback:

```bash
ddsr-bench action=prepare benchmark=physics \
  paths.input=datasets/physics/verified \
  paths.output=tasks/physics/verified
```

These tasks intentionally contain no executable verifier. Official scoring
requires a host-side LLM judge, so Harbor and Docker add no isolation benefit.

## Evaluation

Start or configure two OpenAI-compatible endpoints: one for the model under
test and one for the judge. The bundled vLLM job defaults to ports 8000 and
8001 respectively:

```bash
ddsr-solve --config configs/jobs/physics/vllm.yaml
```

Run one problem by its original ID:

```bash
ddsr-solve --config configs/jobs/physics/vllm.yaml \
  --include-task-name 'Classical Mechanics/2-1'
```

Equivalent `openai.yaml`, `bedrock.yaml`, and `aliyun.yaml` examples configure
both generation and judging. API jobs consume quota. Scores obtained with
different judges are not directly comparable; the published upstream results
use GLM-5.3-Flash after judge self-tests and cross-checks.

Set `agents[0].kwargs.include_figures: false` for the official no-figure subset
used by text-only models: 184 validation and 602 test problems. Leave it true
only when the model endpoint accepts image parts. Static outputs are written
beneath `outputs/static/<job_name>/` and can be resumed with `--resume`.

## Metrics and teacher data

The headline strict accuracy counts a problem correct only when every requested
answer is correct. Collection also reports answer accuracy, mean per-problem
partial credit, RMS calibration error, deterministic bootstrap confidence
intervals, domain accuracy, and answer-type accuracy. Missing or failed
judgments count as wrong.

Use the shared [collection](../../../README.md#-collect-results) and
[teacher-data export](../../../README.md#-export-teacher-data) commands. The
`native` and `full` views preserve the single response-format call, including
public image parts. Provider reasoning remains separate from visible SFT
completion content.

## Compatibility and limitations

The response-format prompt, judge prompt, 1% numeric tolerance, retry behavior,
and metrics match the pinned official harness. Judge output is inherently noisy;
DDSR does not replace the judge with symbolic heuristics or claim comparability
across judge models. DDSR's shared resume policy retains terminal failures;
retry them under a new job output rather than with `--resume`, whereas the
upstream append-only runner retries failed requests when rerun. Automated tests
use fake endpoints and never spend API quota. Exact revisions, prompt hashes,
inference settings, and license are recorded in
[UPSTREAM.md](../../../UPSTREAM.md).
