# MMLU

MMLU evaluates knowledge and reasoning across 57 subjects grouped into STEM,
humanities, social sciences, and other professional domains. DDSR evaluates the
complete 14,042-question official test set rather than extracting a physics or
science subset.

## Data and preparation

```bash
python -m pip install -e '.[mmlu]'
ddsr-bench action=fetch benchmark=mmlu
ddsr-bench action=prepare benchmark=mmlu \
  paths.input=datasets/mmlu/official \
  paths.output=tasks/mmlu/official
```

DDSR reads the official tar archive directly, validates all 57 subjects, and
attaches the first five official development examples to each test question.
The target answer stays verifier-side; demonstration answers are public by
design because they are part of the five-shot prompt.

## Evaluation

```bash
ddsr-solve --config configs/jobs/mmlu/vllm.yaml
ddsr-solve --config configs/jobs/mmlu/vllm.yaml \
  --include-task-name abstract_algebra-0
```

Optional `subjects` and `categories` lists filter the full suite. Static
evaluation is sufficient because responses are parsed as A-D choices and no
model code is executed.

The original 2021 implementation crops prompts to a legacy 2,048-token GPT-2
budget when necessary and scores the next-token log probabilities of
` A`, ` B`, ` C`, and ` D` through the legacy Completions API. DDSR preserves
its five-shot serialization without that obsolete fixed crop and uses generated
answer letters so the same jobs work across vLLM, OpenAI-compatible, Bedrock,
and Aliyun chat endpoints. The result mode is therefore explicitly
`mmlu-generative`; it must not be presented as byte-identical likelihood-based
leaderboard evaluation.

## Scores and trajectories

Collection reports fixed-denominator accuracy overall, by attempt, category,
and subject. Malformed generated answers receive zero. The `native` and `full`
trajectory views preserve the exact five-shot request and model response without
the target answer key.

See the [pipeline](PIPELINE.md), [dataset catalog](../../../DATASETS.md), and
[upstream record](../../../UPSTREAM.md) for source and correction limitations.
