# CritPt-AI grading

Assess saved Auto/Freestyle responses against private canonical references.
This is the CritPt-AI grader incorporated from DDPR, not a shared physics scorer
or a replacement for consensus. See [UPSTREAM.md](UPSTREAM.md) for provenance.

## Python interface

```python
from ddsr_bench.benchmarks.critpt.data.loader import load_ai_task
from ddsr_bench.benchmarks.critpt.evaluation.ai import grade_response

problem = load_ai_task("/path/to/compiled-task")
report = grade_response(problem, student_response, attempt_id="trial-1")
print(report.grading_status)
```

Both DDSR-Bench and DDPR use the unified CritPt data loader for prepared tasks.
`load_ai_task` restores public and private fields from `instruction.md` and
`tests/reference.json`, without executing code or reading the original corpus.
The compatibility parser `parse_grading_problem` accepts native Auto/Freestyle
records only. References and reports are private verifier data.
Keep them out of student prompts and public generation artifacts.

Rule grading supports floating, exact, symbolic and categorical answers using
an `answer()` function, including upstream composite/multi-target grading.
Freestyle assesses natural-language responses against the reference answer and
solution. Existing schemas, report fields and prompt semantics are preserved.
The nested `io/` directory handles native record compatibility and report output,
not prepared-task loading.

The report distinguishes `pass`, `fail`, invalid answers, resource limits and
grader failures. It does not assign a training reward. DDPR owns that mapping.
The answer restrictions and deadlines are not container isolation; do not run
untrusted corpus functions on the host. Static generation does not invoke the
grader automatically. `ai.docker.execute(code, resources=...)` provides the
Harbor-backed execution callback: pass it with bound resources as
`grade_response(..., execute=...)`. Only code enters the container, and bounded
JSON values return; references and judge credentials stay outside. Build the
current CritPt image before using it.

## Submission

To generate and assess in one command, add this optional block to your existing
one-step AI generation job YAML:

```yaml
submission:
  backend: internal
  attempts: [0] # Explicit attempt indices, not a count.
  ai:
    llm: {} # SemanticJudgeConfig overrides; separate from the teacher model.
    resources: # Required for rule answers; Docker image must be built first.
      image: ddsr-bench-critpt:latest
      cpus: 2
      memory_mb: 4096
      timeout_sec: 120
    llm_fallback: false # Allow LLM equivalence checking when deterministic grading is inconclusive.
```

Run the unchanged command: `ddsr-solve --config configs/local/critpt-ai.yaml`.
Without `submission`, behavior remains generation and static checks only.
With it, the runner finishes generation, collects results, then grades the selected attempts.
It does not grade trials as they finish or regenerate rejected answers. Configure
the judge and worker resources below; rule assessment requires the current Docker
image. If grading fails, preserve the output and use `action=submit` to retry
assessment without generation (reconcile any pending claim first).

After generation, explicitly assess selected attempts of rule- or LLM-graded tasks:

```bash
ddsr-bench action=submit paths.input=JOB_DIR \
  benchmark.submission.backend=internal benchmark.submission.attempts=0
```

Submission detects the source from collected trials and reuses `datasets[].path`
from the job's saved `job-config.yaml` (or Harbor `config.json`). Relative paths
use the current working directory, as during generation. If data moved or the
saved config is missing, supply
`'benchmark.submission.datasets=[{path:/absolute/path/to/tasks}]'`.
Dataset roots contain task directories, matching generation; no recursive scan.

Each task's `grader` defaults to rule or LLM assessment. Set
`benchmark.submission.grader=rule` or `llm` to override it for this submission;
the task must supply that grader's required reference fields. Configure the LLM client
through `benchmark.submission.ai.llm` (`SemanticJudgeConfig` fields), for example
`+benchmark.submission.ai.llm.model=MODEL`; this cannot override the task's grader.
Both read the saved raw response, regardless of static format-check results.
Rule grading extracts the last Python/py block defining `answer()`; LLM grading
assesses the full response. Neither relies on the static `answer.py` artifact.
Empty settings retain the imported client defaults below, not the generation
model's settings. Rule workers use `benchmark.submission.ai.resources`, defaulting
to the configured Harbor image, CPU, memory and timeout limits, with no host
execution fallback. Each answer runs in a fresh container. Optional rule-equivalence
LLM fallback when deterministic grading is inconclusive is off by default
(`ai.llm_fallback=false`); enabling it uses the
equivalence judge environment settings below, not `ai.llm`.
All selected tasks, references and saved responses are checked before judge calls.
The private `submission-internal-ai-0.json` (or `-rule-0.json` / `-llm-0.json`
for explicit overrides) preserves task IDs, original corpus IDs,
attempts and detailed reports. Generation records and export filters are unchanged.
An interrupted submission leaves a `.pending.json` claim; reconcile it before
retrying because some assessments may already have been billed.

## Judge configuration

Defaults retain the standalone grader's Bedrock Responses configuration:
`openai.gpt-5.6-sol`, high reasoning, `us-east-1`. Configure model availability
and credentials in the worker environment, never in problem metadata.

- Equivalence fallback prefix: `AUTO_CRITPT_GRADER_JUDGE_`.
- Semantic judge prefix: `AUTO_CRITPT_GRADER_SEMANTIC_JUDGE_`.
- Suffixes: `BACKEND`, `MODEL`, `BASE_URL`, `API_KEY_ENV`, `REGION`,
  `MAX_TOKENS`, `REASONING_EFFORT`, `TIMEOUT_SECONDS`, `HTTP_MAX_ATTEMPTS`,
  `HTTP_BACKOFF_SECONDS`, `EXTRA_BODY_JSON`.
- `BACKEND=ddsr` adds `PROVIDER`: `vllm`, `openai`, `aliyun`, or `bedrock`.
  DDSR-Bench owns requests/retries; `HTTP_BACKOFF_SECONDS` is unused and nonempty
  `EXTRA_BODY_JSON` is rejected. JSON judgments are requested in the prompt.
- `BACKEND=bedrock` retains Responses; `BACKEND=ddsr` with `PROVIDER=bedrock`
  uses Chat Completions. Native Bedrock credentials use `AWS_BEDROCK_API_KEY`
  (or `AWS_BEARER_TOKEN_BEDROCK`). Other paths use `API_KEY_ENV`.
- `AUTO_CRITPT_GRADER_TIMEOUT_SECONDS` bounds local grading (default 120 seconds);
  `AUTO_CRITPT_NUMERIC_CHECK_TIMEOUT_SECONDS` bounds numeric evidence (30 seconds).
- `AUTO_CRITPT_GRADER_JUDGE_MIN_CONFIDENCE` defaults to 0.90.

Both judges accept a synchronous `completion(messages, config)` callable for
injection. The normal `complete_judge` dispatcher selects the configured backend;
DDSR-Bench async clients are bridged and closed on each call. Visible response
content is parsed separately from reasoning. Pass `llm_fallback=False` to
`grade_response` to disable rule fallback; semantic grading still needs a judge.

API migration: external Auto/Freestyle callers must replace `judge_enabled=`
with `llm_fallback=`. The old keyword is no longer accepted. The Python default
remains `True`; submission configuration still defaults to `False`. Input and
report formats are unchanged.

## Verification

```bash
python -m pytest -q tests/benchmarks/critpt/ai
# Optional synthetic execution check after building the current CritPt image:
DDSR_TEST_AI_IMAGE=ddsr-bench-critpt:latest \
  python -m pytest -q tests/benchmarks/critpt/ai/test_docker.py -k live
# Mixed rule/LLM submission: real Docker worker, mocked semantic judge:
DDSR_TEST_AI_IMAGE=ddsr-bench-critpt:latest \
  python -m pytest -q tests/benchmarks/critpt/ai/test_collection.py -k mixed_graders
```

Core tests cover rules, validation, deadlines, provider errors and mocked judgments.
For differential checks, set `DDSR_TEST_AUTO_CRITPT` to an upstream checkout at
the commit pinned in [UPSTREAM.md](UPSTREAM.md), then run
`python -m pytest -q tests/benchmarks/critpt/ai/test_parity.py`.
DDPR retains reward mapping, compiled-task and saved-judgment replay checks.
Neither unit tests nor replay establish live model quality or execution isolation.
