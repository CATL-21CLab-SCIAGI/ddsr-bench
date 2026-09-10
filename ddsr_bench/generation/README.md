# Generation

ddsr-bench supports local and hosted Chat Completions clients. Use `ddsr-smoke`
to check an endpoint, then pass the same settings to a benchmark job.

Harbor job examples live in `configs/jobs/BENCHMARK`. Their `agents[].kwargs`
sections make the endpoint, strategy, streaming behavior, and request-time
sampling parameters explicit. Run one with
`harbor run --config configs/jobs/BENCHMARK/CLIENT.yaml`.

Shared API clients live in `ddsr_bench/generation`. Benchmark prompt and
conversation behavior lives in each `benchmarks/BENCHMARK/generation` package;
Harbor and static evaluation adapters live beside one another under
`benchmarks/BENCHMARK/evaluation`.

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

PAI Token Service exposes the same OpenAI-compatible routes, so it reuses
`OpenAIClient`. Bundled benchmark `aliyun.yaml` jobs target the Beijing endpoint
and `qwen3.8-max`:

```bash
export ALIYUN_API_KEY='...'
ddsr-smoke \
  --client openai \
  --base-url https://cn-beijing.pai-token.aliyuncs.com/v1 \
  --api-key-env ALIYUN_API_KEY \
  --model qwen3.8-max
```

For another region, update both `base_url` and `extra_allowed_hosts` in the job
file. The API key is read at runtime and is never stored in the configuration.

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
