# CMT-Benchmark pipeline

CMT-Benchmark is a text-scoring benchmark. Direct static evaluation is
preferred; optional Harbor normalization provides the same task and artifact
boundaries as the other supported benchmarks.

```mermaid
flowchart TD
    A["Pinned v2 dataset: 50 problems"] --> B["Project public type and prompt"]
    B --> C{"Evaluation path"}
    C -->|"direct"| D["ddsr-solve; no preparation or Docker"]
    C -->|"optional"| E["Normalize as Harbor task"]
    E --> F["Harbor agent writes answer.txt"]
    D --> G["Extract final boxed expressions"]
    F --> G
    G --> H{"Reference contract"}
    H -->|"numeric"| I["Numeric comparison"]
    H -->|"parameters"| J["Symbolic comparison"]
    H -->|"functions or NC"| K["Functional comparison"]
    I --> L["Binary reward"]
    J --> L
    K --> L
    L --> M["Collect results or export trajectory"]
```

One trial is one problem-attempt pair. Concurrency schedules independent trials
without changing attempt grouping. Multiple answers are separated by semicolons
inside one final box. Missing boxes, malformed expressions, worker errors, and
timeouts receive zero; no repair prompt is sent.

Pass rate uses all scored problems as its denominator. Parse rate counts trials
whose expressions were evaluated successfully, whether equivalent or different.
