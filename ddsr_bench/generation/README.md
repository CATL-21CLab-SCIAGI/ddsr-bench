# Generation

ddsr-bench supports local and hosted Chat Completions clients. Use `ddsr-smoke`
to check an endpoint, then pass the same settings to a benchmark job.

Job examples live in `configs/jobs/BENCHMARK`. Use `ddsr-solve --config` for
supported static runs, or `harbor run --config` when isolated execution is needed.
Static CritPt generation still produces Python answers; it validates their
structure without executing them or establishing their correctness.

Shared API clients live in `ddsr_bench/generation`. Benchmark prompt and
conversation behavior lives in each `benchmarks/BENCHMARK/generation` package;
Harbor and static evaluation adapters live beside one another under
`benchmarks/BENCHMARK/evaluation`.

## Generation quick start

After [installation](../../README.md#-installation) and benchmark preparation,
activate your environment and work from the repository root. This example uses
[prepared CritPt tasks](../benchmarks/critpt/README.md#data-and-preparation)
and PAI, without Docker. Export the configured API key and check access using
the [provider example](#alibaba-cloud-pai-token-service) below; `.env` files are
not loaded automatically. Model requests consume API quota.

```bash
mkdir -p configs/local
cp configs/jobs/critpt/aliyun.yaml configs/local/critpt-check.yaml
```

In the copy, set `job_name: critpt-check`, one attempt, and concurrency one.
Check the dataset path, model, endpoint, credential variable, and sampling limits.
Then test one task:

```bash
ddsr-solve --config configs/local/critpt-check.yaml \
  --include-task-name Challenge_1_main
```

Inspect `agent/response.json`, `artifacts/answer.py` (if valid), and
`validation/result.json` under
`outputs/static/critpt-check/Challenge_1_main__attempt-0/`.
Validation does not establish answer correctness.

Copy the config to `configs/local/critpt-batch.yaml`, set `job_name: critpt-batch`
to avoid colliding with the check, and choose the attempt count and concurrency:

```bash
# Generate the full batch.
ddsr-solve --config configs/local/critpt-batch.yaml
# Only if interrupted: resume with unchanged generation settings and task selection.
ddsr-solve --config configs/local/critpt-batch.yaml --resume
# Once complete: summarize without model calls.
ddsr-bench action=collect paths.input=outputs/static/critpt-batch
```

With the default `jobs_dir: outputs/harbor`, static outputs go to
`outputs/static/JOB_NAME`. Relative paths resolve from the working directory;
use persistent output storage on remote machines. See below for
[resume](#resuming-static-jobs) and [long-context settings](#critpt-long-context-generation),
or use the [direct JSON example](../benchmarks/critpt/README.md#solve-one-challenge)
to skip task preparation.

## vLLM

`VLLMClient` checks `/v1/models` and supports vLLM options such as `top_k` and
chat-template thinking controls.

```bash
ddsr-vllm configs/serving/vllm/macos-qwen38.yaml
ddsr-smoke \
  --client vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model ddsr-local
```

The native YAML determines the model path and served model name. See the
[vLLM server documentation](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).

## OpenAI

`OpenAIClient` uses the OpenAI request schema and checks `/v1/models`.

```bash
export OPENAI_API_KEY='...'
ddsr-smoke \
  --client openai \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-5.6-luna
```

See the
[OpenAI Chat Completions reference](https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions).

### Alibaba Cloud PAI Token Service

PAI Token Service exposes OpenAI-compatible routes. `AliyunClient` preserves its
documented `max_tokens`, `temperature`, `top_p`, `seed`, and `enable_thinking`
fields. Bundled `aliyun.yaml` jobs target the Beijing endpoint and
`qwen3.8-max`:

```bash
export ALIYUN_API_KEY='...'
ddsr-smoke \
  --client aliyun \
  --base-url https://cn-beijing.pai-token.aliyuncs.com/v1 \
  --api-key-env ALIYUN_API_KEY \
  --model qwen3.8-max
```

For another region, update both `base_url` and `extra_allowed_hosts` in the job
file. The API key is read at runtime and is never stored in the configuration.

`reasoning_effort` is also forwarded when explicitly set; support and accepted
values depend on the served model. It remains mutually exclusive with
`enable_thinking`. There is no automatic reasoning downgrade.

`agents[].kwargs.request_profile` selects the request fields used by
`AliyunClient`, not the provider or model:

- `native` (default): maps `sampling.max_tokens` to API field `max_tokens`,
  sends `temperature` and `top_p`, and supports
  `enable_thinking` or `reasoning_effort` when configured.
- `openai`: maps `sampling.max_tokens` to API field `max_completion_tokens`
  and sends optional `reasoning_effort`;
  omits `temperature` and `top_p`, and rejects `enable_thinking`.

Always use `sampling.max_tokens` in job YAML; do not add a
`max_completion_tokens` config key. For example, `max_tokens: 393216` becomes
`"max_completion_tokens": 393216` in an `openai`-profile request.

Both use the configured Chat Completions endpoint and API key. Keep `native`
unless the endpoint requires the other format or you are preserving an older
job's request behavior, as in the DeepSeek example.
CritPt sends its seed with either profile; whether PAI DeepSeek honors it remains
unverified. Direct `openai`-profile client calls must pass `chat(seed=...)`.

`sampling` records configured settings; each response's `request_parameters`
records sent generation fields. Omitted non-default settings trigger a warning.
This metadata is unavailable for older records.

## Resuming static jobs

Use `ddsr-solve --config JOB.yaml --resume` for CritPt, CMPhysBench, and PHYBench.
Completed trials, including recorded failures, are reused; this is not a retry
failed trials option. Keep job settings and selected inputs unchanged, except
for concurrency. SciCode has no static runner; its execution remains in Harbor.
For SciCode, use [Harbor job recovery](../benchmarks/scicode/README.md#recovery).

CMPhysBench and PHYBench save `resume.json` with job settings and an input hash.
Jobs predating that snapshot cannot be resumed safely. Interrupted trial folders
are preserved under `.interrupted/` before the whole trial restarts. This may
repeat a model request if its response was not saved as a completed trial.
CritPt additionally reuses completed generation stages, as described below.

## CritPt long-context generation

Use `ddsr-solve --config` with `configs/jobs/critpt/vllm-long-context.yaml` or
`configs/jobs/critpt/aliyun-deepseek.yaml`. Both default to one attempt and
concurrency four; check token budgets against your endpoint's limits.

- **Budgets:** `sampling.max_tokens` caps derivation; `formatting_max_tokens`
  caps the second, formatting call. With vLLM, `context_window` uses `/tokenize`
  to subtract input length and `context_safety_tokens` (default 32). It does not
  extend the server's context. Set `require_complete_stages: true` to reject
  empty or non-stop completions instead of continuing to formatting.
- **Seeds:** `seed_base: 42` derives a stable seed per problem-attempt, shared
  by both calls; provider determinism is not guaranteed. This is static-only.
  For Harbor, remove it and use `sampling.seed` for a fixed seed across trials.
- **Recovery:** `--resume` reuses completed trials and `agent/stage-N.json`
  checkpoints. Streaming also saves partial output in `stage-N.stream.jsonl`.
  Keep generation settings unchanged; only concurrency may change. Retrying
  with a different budget requires a separate run, not overwriting old results.

<details>
<summary>Troubleshooting: resuming an older or relocated CritPt job</summary>

Use `resume_migration` only when an older run's output or dataset paths moved:

```yaml
resume_migration:
  path_prefixes:
    /old/critpt-eval/outputs: /shared/critpt/runs/outputs
    /old/tasks: /shared/critpt/tasks
```

Compatibility also covers former CritPt agent names and the transition from
`client_name: openai` to `aliyun` with `request_profile: openai`. It does not allow
changes to model, endpoint, prompts, seeds, or budgets. Original `job-config.yaml`
is preserved; mappings are recorded separately. Omit this setting for normal runs.

</details>

## Amazon Bedrock

`BedrockClient` uses Bedrock's OpenAI-compatible request format. Replace the
region and model with ones available to your account.

```bash
export AWS_BEARER_TOKEN_BEDROCK='...'
ddsr-smoke \
  --client bedrock \
  --base-url https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1 \
  --api-key-env AWS_BEARER_TOKEN_BEDROCK \
  --model global.openai.gpt-5.6-luna
```

On `bedrock-runtime`, the `global.` prefix selects Bedrock's system-defined
inference profile; this Luna model rejected direct on-demand invocation there.
The `bedrock-mantle` endpoint accepts the bare `openai.gpt-5.6-luna` model ID.

See the
[Amazon Bedrock endpoint documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html).
