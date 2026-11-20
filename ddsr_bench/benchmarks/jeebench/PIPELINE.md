# JEEBench pipeline

```mermaid
flowchart TD
    A["Pinned official data.zip"] --> B["515 validated questions"]
    B --> C{"Prompt mode"}
    A -->|"one-shot only"| C
    C --> D["Shared chat client"]
    D --> E["Detailed solution and final answer"]
    E --> F["DDSR final-answer extraction"]
    F --> G{"Official response-type scorer"}
    G -->|"MCQ"| H["Exact choice"]
    G -->|"Multiple MCQ"| I["Exact or subset partial credit"]
    G -->|"Integer / Numeric"| J["Absolute tolerance 0.01"]
    H --> K["Paper, subject, and type summaries"]
    I --> K
    J --> K
    E --> L["Reference-free trajectory export"]
```

Static evaluation is complete because model text is parsed but never executed.
One trial is one question and one attempt; multiple attempts can reproduce
self-consistency sampling without changing the benchmark scorer.
