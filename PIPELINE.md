# Framework pipeline

Every benchmark supplies adapters for its source data, generation policy,
verifier, result records, and trajectories. The shared framework supplies model
clients, job configuration, collection, and training-data export.

```mermaid
flowchart TD
    A["Benchmark source data"] --> B["Benchmark data adapter"]
    B --> C["Prepared task directories"]
    C --> D["Job configuration"]
    D --> E{"Evaluation mode"}
    E -->|"Harbor"| F["Schedule problem × attempt"]
    E -->|"Benchmark supports static mode"| G["Schedule problem × attempt"]
    F --> H["Benchmark generation adapter"]
    G --> H
    H --> I["Shared model client"]
    I --> J["Generated benchmark artifact"]
    J --> K{"Verifier"}
    K -->|"Harbor"| L["Isolated benchmark execution"]
    K -->|"Static"| M["Non-executing validation"]
    L --> N["Trial result"]
    M --> N
    N --> O["Benchmark result adapter"]
    O --> P["summary.json + summary.csv"]
    N --> Q["Benchmark trajectory adapter"]
    Q --> R["trajectories.jsonl + sft.jsonl"]
```

A trial is one evaluation of one problem for one attempt. Concurrency controls
how many trials run at once; it does not combine them or cause their files to
share a directory.

The benchmark pipelines define what one generated artifact contains and how it
is verified:

- [CritPt pipeline](ddsr_bench/benchmarks/critpt/PIPELINE.md)
- [SciCode pipeline](ddsr_bench/benchmarks/scicode/PIPELINE.md)
