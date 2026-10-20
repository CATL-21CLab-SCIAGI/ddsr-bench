# CritPt-AI grading

Assess saved Auto/Freestyle responses against private canonical references.
This is the CritPt-AI grader incorporated from DDPR, not a shared physics scorer
or a replacement for consensus. See [UPSTREAM.md](UPSTREAM.md) for provenance.

## Python interface

```python
from ddsr_bench.benchmarks.critpt.evaluation.ai import (
    grade_response,
    load_problem,
    load_problems,
    parse_grading_problem,
)

problem = load_problem("/path/to/compiled-task")
report = grade_response(problem, student_response, attempt_id="trial-1")
print(report.grading_status)
```

`load_problem` reads `instruction.md` and `tests/reference.json` without executing
code or reading the original corpus. `load_problems` takes explicitly selected
task paths in caller order. `parse_grading_problem` accepts native Auto/Freestyle
records or validated models. References and reports are private verifier data.
Keep them out of student prompts and public generation artifacts.

Rule grading supports floating, exact, symbolic and categorical answers using
an `answer()` function; composite/multi-target grading remains unsupported.
Freestyle assesses natural-language responses against the reference answer and
solution. Existing schemas, report fields and prompt semantics are preserved.
The nested `io/` directory retains the external I/O adapter structure.

The report distinguishes `pass`, `fail`, invalid answers, resource limits and
grader failures. It does not assign a training reward. DDPR owns that mapping.
The answer restrictions and deadlines are not container isolation; do not run
untrusted corpus functions on the host. This incorporation exposes the Python
grader only. Static generation and submission do not invoke it automatically;
Harbor/submission integration remains separate work.

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
content is parsed separately from reasoning. Pass `judge_enabled=False` to
`grade_response` to disable rule fallback; semantic grading still needs a judge.

## Verification

```bash
python -m pytest -q tests/benchmarks/critpt/ai
```

Core tests cover rules, validation, deadlines, provider errors and mocked judgments.
DDPR retains reward mapping, compiled-task and saved-judgment replay checks.
Neither unit tests nor replay establish live model quality or execution isolation.
