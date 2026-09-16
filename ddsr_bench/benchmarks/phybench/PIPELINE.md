# PHYBench pipeline

PHYBench is a text-scoring benchmark. Direct static evaluation is preferred;
optional Harbor normalization provides the same task and artifact boundaries as
the other supported benchmarks.

```mermaid
flowchart TD
    A["Pinned canonical file: 500 problems"] --> B{"Official answer present?"}
    B -->|"yes: 100"| C["Build public tag and question"]
    B -->|"no: 400"| D["Generation-only; omit from local scoring"]
    C --> E{"Evaluation path"}
    E -->|"direct"| F["ddsr-solve; no preparation or Docker"]
    E -->|"optional"| G["Normalize as Harbor task"]
    G --> H["Harbor agent writes answer.txt"]
    F --> I["Extract final balanced boxed formula"]
    H --> I
    I --> J["Process-isolated EED worker"]
    J --> K["EED score 0–100"]
    K --> L["Normalize reward to 0–1"]
    L --> M["Collect results or export trajectory"]
```

One trial is one problem-attempt pair. Concurrency schedules independent trials
without changing attempt grouping. A missing or malformed final box receives
zero, and no repair prompt is sent.

EED compares symbolic expression trees and applies discounted edit costs for
clustered differences. A score of 100 is counted as exact accuracy; the shared
reward is `eed_score / 100`.
