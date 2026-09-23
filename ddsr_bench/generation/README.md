# Generation

ddsr-bench supports local and hosted Chat Completions clients. Use `ddsr-smoke`
to check an endpoint, then pass the same settings to a benchmark job.

Harbor job examples live in `configs/jobs/BENCHMARK`. Their `agents[].kwargs`
sections make the endpoint, strategy, streaming behavior, and request-time
sampling parameters explicit. Run one with
`harbor run --config configs/jobs/BENCHMARK/CLIENT.yaml`.
The CritPt `qwen-long-context.yaml` and `deepseek-pai-max.yaml` examples use
static-only `seed_base`: run them with `ddsr-solve --config`, or remove that field
and choose a fixed `sampling.seed` before running them with Harbor.

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

For CritPt jobs migrated from an OpenAI-compatible PAI client, set
`client_name: aliyun` and `request_profile: openai` in `agents[].kwargs`.
That profile sends `max_completion_tokens` and `reasoning_effort`, omitting
`temperature` and `top_p` just as the previous OpenAI client did. The default
`native` profile retains DDSR's `max_tokens`, `temperature`, and `top_p` body.
Both use the same Chat Completions endpoint and the configured `api_key_env`;
neither switches to Responses API.

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

CritPt static jobs can connect directly to an already running vLLM server.
Set `context_window` and optional `context_safety_tokens` (default 32) to count
each stage's input through `/tokenize` and cap output at the remaining context.
No context extension is applied by the client. `sampling.max_tokens` caps stage
one; `formatting_max_tokens` independently caps stage two. Prompt text and the
two-step conversation remain unchanged, including formatting after an empty or
length-limited first-stage content response. `require_complete_stages: true`
is an explicit stricter alternative, disabled by default.

The static runner refills each free concurrency slot immediately. `seed_base`
derives a stable seed from `(base, problem_id, attempt)` and sends the same seed
to both stages. Seeds do not imply deterministic GPU kernels or provider
behavior. This option requires the static runner; Harbor can use a fixed
`sampling.seed` with a supporting client.

Each completed stage is saved atomically under `agent/stage-N.json`. Streaming
requests also retain `stage-N.stream.jsonl`, including partial output after a
disconnect. Interrupted answers are not silently retried after receiving a
stream chunk. `ddsr-solve --config JOB.yaml --resume` reuses completed trials
and validated stage checkpoints; only concurrency may change in a normal
resume. Increasing a formatting budget requires a separate retry directory
with only the saved first stage copied, then `run_trial(..., resume=True)`.
Never overwrite a completed trial to retry it without first preserving its
original artifacts and provenance.

Injected clients with the original `chat(messages)` interface remain supported;
their sampling settings remain client-managed. Per-trial `seed_base` and an
explicit formatting budget require `seed` and `max_tokens` keyword support,
respectively, and fail explicitly when unsupported. Stream journals require
`stream_path` support; stage checkpoints do not.

### Legacy resume compatibility

CritPt's optional `resume_migration` setting supports historical jobs with moved
storage or former agent/client names. Omit it for new jobs; it is not required
for normal `--resume` or generation-stage checkpoint recovery.
Declare mappings in the job YAML alongside the normal job settings:

```yaml
resume_migration:
  path_prefixes:
    /old/critpt-eval/outputs: /shared/critpt/runs/outputs
    /old/tasks: /shared/critpt/tasks
```

Mappings apply only to job output and dataset paths. The compatibility check
also recognizes the former CritPt agent namespace and the equivalent transition
from `client_name: openai` to `client_name: aliyun` with `request_profile: openai`.
It still rejects changes to the model, endpoint, prompts/strategy, seed, sampling,
stage budgets, or other generation settings. Stage reuse separately validates
the saved public problem, messages, and sampling.

Mapped configuration is saved as a new resume audit, never over the original
`job-config.yaml`. Preserve original trials and configs when relocating runs.
This compatibility option remains necessary only for jobs using those old paths
or names; it does not relax the retry restrictions described above.

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
