# Upstream grading snapshot

Source: `CATL-21CLab-SCIAGI/Auto_CritPt_Grader`, Apache-2.0.
Commit: `565d1b0bfab353e17a40a38fe656c3ab0caee660` (version 0.6.0).
Original directory: `src/auto_critpt_grader/`.
The accompanying LICENSE is copied from that repository.

The standard grading core was incorporated via DDPR commit `e7c04fc`.
Upstream extraction, deterministic scoring, composite composition, judge prompts,
confidence policy, JSON selection and failure classifications are the behavior
contract. Earlier DDPR changes to these rules have been reverted for parity.
This deliberately preserves upstream edge cases: sampled symbolic constancy,
submitted symbol assumptions, order-dependent aliases, weight clamping, and
reference/comparison failures that upstream classifies as failed grades.
Changing those semantics requires a separately approved upstream fix.

## Integration adapters

- Native Auto/Freestyle input and reports live in `io/`; normalized prepared tasks
  enter through CritPt's `data` loaders. `grader.py` owns orchestration.
- Upstream `primitives.py` is split into `rule/`, `answer.py`, `validation.py` and
  `schemas.py`. Existing DDSR function names and signatures are retained, including
  `(answer, target, **kwargs)` comparisons and `grade_exact_numeric`.
- `llm/` owns assessment policy; `llm/clients/` retains OpenAI/Bedrock transports
  and adds DDSR client support. Both judges accept `(messages, config)` completion
  callbacks; semantic assessment receives explicit public/reference fields.
  Diagnostic names may reflect these local APIs rather than upstream class names.
- Python `llm_fallback` replaces upstream `judge_enabled` for Auto equivalence
  fallback only. Disabling it does not disable Freestyle's semantic judge.
  Submission defaults it to false; the Python API defaults it to true.
- DDSR reports omit upstream `schema_version` and derived `reward`. DDPR owns
  reward mapping, including its unresolved resource-limit policy. Grading evidence,
  scores and statuses remain in the shared report.
- Submission can inject Harbor-isolated answer execution. Only answer code enters
  workers; references remain host-side. The Python default remains local execution.
  No solver, batch runner, per-problem grader generator or standalone CLI is copied.

## Execution and transport safeguards

Finite timeout/backoff checks, reliable signal cleanup, malformed HTTP-envelope
handling and configured retry counts remain integration safeguards. These are not
claims of identical transport behavior. They must not alter deterministic scoring;
nonfinite deadlines are rejected rather than weakening execution limits. No AST
execution restriction was relaxed for parity. No live-provider equivalence is claimed.

## Differential verification

Check out the pinned commit separately and run:

```bash
DDSR_TEST_AUTO_CRITPT=/path/to/Auto_CritPt_Grader \
  python -m pytest -q tests/benchmarks/critpt/ai/test_parity.py
```

The fixture verifies the commit before importing upstream. Tests compare trusted
synthetic values/responses, grading plans, reports and mocked judge messages/results.
They make no paid calls and execute no corpus code. Without the optional checkout,
these tests skip; ordinary regressions still run. Re-run DDPR's reward tests when
changing this boundary. Do not infer live model quality or GPU-training parity
from these checks.
