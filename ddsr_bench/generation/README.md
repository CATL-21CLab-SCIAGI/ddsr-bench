# Generation

ddsr-bench supports local and hosted Chat Completions clients. Use `ddsr-smoke`
to check an endpoint, then pass the same settings to a benchmark job.

Job examples live in `configs/jobs/BENCHMARK`. Use `ddsr-solve --config` for
supported static runs, or `harbor run --config` when isolated execution is needed.
Benchmark guides describe their output formats, validation, and recovery behavior.

Shared API clients live in `ddsr_bench/generation`. Benchmark prompt and
conversation behavior lives in each `benchmarks/BENCHMARK/generation` package;
Harbor and static evaluation adapters live beside one another under
`benchmarks/BENCHMARK/evaluation`.

## Client setup

After [installation](../../README.md#-installation), activate your environment
and run the provider's smoke example below. Export the configured API key;
`.env` files are not loaded automatically. Model requests consume API quota.

Use the same client, endpoint, model, and credential variable in your benchmark's
job config. See the [benchmark catalog](../../BENCHMARKS.md) for task preparation,
single-problem runs, batch generation, and benchmark-specific settings.

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

PAI Token Service exposes OpenAI-compatible routes. Bundled `aliyun.yaml` jobs
use `AliyunClient`, the Beijing endpoint, and `qwen3.8-max`:

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

Set `agents[].kwargs.request_profile` to select the API fields:

- `native` (default): maps `sampling.max_tokens` to API field `max_tokens`,
  sends `temperature` and `top_p`, and supports
  `enable_thinking` or `reasoning_effort` when configured.
- `openai`: maps `sampling.max_tokens` to API field `max_completion_tokens`
  and sends optional `reasoning_effort`;
  omits `temperature` and `top_p`, and rejects `enable_thinking`.

Always use `sampling.max_tokens` in YAML; the profile translates it. Check limits
for the selected API field, not just the model's context size. Reasoning options
depend on the model; `reasoning_effort` and `enable_thinking` are mutually exclusive.
Provider determinism is not guaranteed.
Direct `openai`-profile client calls must pass `chat(seed=...)` to send a seed.

`sampling` records configured settings; each response's `request_parameters`
records sent generation fields. Omitted non-default settings trigger a warning.
This metadata is unavailable for older records.

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

## Transport and recovery

`timeout` limits network operations; optional `read_timeout` overrides read
inactivity. Neither is a total generation deadline. Read/write/protocol failures
are not automatically retried because the provider may already be generating
or charging for the request.

For supported static jobs, `ddsr-solve --config JOB.yaml --resume` reuses completed
trials, including recorded failures; it is not a retry-failures option. Keep
settings and inputs unchanged except for concurrency. Check the benchmark guide
for checkpoint behavior; incomplete trials may repeat a model request.
