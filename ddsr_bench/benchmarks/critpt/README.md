# CritPt

CritPt challenges contain one main problem and optional indexed subproblems.
ddsr-bench preserves the pinned official prompts and supports both official
generation strategies:

- one-step: derive the result and return code in one model call;
- two-step: derive first, then use CritPt's exact formatting prompt to produce
  code without further reasoning.

The benchmark package is arranged by capability:

```text
data/          challenge loading and public/private projections
generation/    official prompts and conversation runner
evaluation/    Harbor/static modes, preparation, verification, submission
result.py      adapter for shared result collection
trajectory.py  adapter for shared training-data export
```

Use the root [README](../../../README.md) for the complete workflow and
[PIPELINE](../../../PIPELINE.md) for the execution flow. Client setup and direct
single-problem generation are documented in the shared
[generation guide](../../generation/README.md).

Static evaluation validates generated Python without executing it. Harbor adds
container isolation and reference/testcase execution when those private fields
are available. Official public tasks therefore receive format validation until
their complete 70-problem batch is explicitly sent to Artificial Analysis.

Artificial Analysis submissions retain CritPt's official wire fields:
`problem_id`, `generated_code`, `model`, `generation_config`, and `messages`.
Shared collection terminology such as `artifact` does not alter that payload.
