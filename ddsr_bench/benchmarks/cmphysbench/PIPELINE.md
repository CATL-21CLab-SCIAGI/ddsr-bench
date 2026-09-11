# CMPhysBench pipeline

CMPhysBench is a text-scoring benchmark. Direct static evaluation is preferred
because the model output is never executed. Optional Harbor normalization is
available for consistency with execution-based benchmarks.

```mermaid
flowchart TD
    A["Pinned Hugging Face dataset"] --> B["Load all 520 records"]
    B --> C{"Official final_answer present?"}
    C -->|"yes"| D["Build public question and symbols"]
    C -->|"no"| E["Report skipped"]
    D --> F{"Evaluation path"}
    F -->|"direct"| G["ddsr-solve; no preparation or Docker"]
    F -->|"optional"| M["Normalize as Harbor task"]
    M --> N["Harbor agent writes answer.txt"]
    G --> H["Extract first balanced boxed answer"]
    N --> H
    H --> I["Timeout-controlled SEED worker"]
    I --> J["SEED score 0–100"]
    J --> K["Normalize reward to 0–1"]
    K --> L["Collect results or export trajectory"]
```

One trial is one problem-attempt pair. Concurrency schedules independent trials
without changing attempt grouping. Missing or malformed boxed answers receive
zero; the evaluator never sends a repair prompt.

SEED supports Expression, Equation, Tuple, Interval, and Numeric answers. The
adapted implementation preserves the official symbolic equivalence, tree-edit
partial credit, tuple recursion, interval handling, and numeric tolerances.
