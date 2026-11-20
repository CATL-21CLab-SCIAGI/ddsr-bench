# Framework pipeline

Every benchmark supplies adapters for its source data, generation policy,
verifier, result records, and trajectories. The shared framework supplies model
clients, job configuration, collection, and training-data export.

```mermaid
flowchart TD
    A["Pinned benchmark source"] --> B["Shared source catalog"]
    B --> C0["Validated local edition"]
    C0 --> B0["Benchmark data adapter"]
    B0 --> C["Prepared task directories"]
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
    N --> S["Explicit submission, when supported"]
    S --> T["Benchmark official or internal grader"]
    T --> U["Separate submission report"]
```

A trial is one evaluation of one problem for one attempt. Concurrency controls
how many trials run at once; it does not combine them or cause their files to
share a directory.

Submission is explicit and never runs automatically after generation. CritPt's
internal backend uses an external evaluator-only reference bundle and isolated
execution. Its report does not change trial rewards or training-data selection.
The compatibility CLI also supports multi-attempt aggregation and reference replay.

The benchmark pipelines define what one generated artifact contains and how it
is verified:

- [CritPt pipeline](ddsr_bench/benchmarks/critpt/PIPELINE.md)
- [SciCode pipeline](ddsr_bench/benchmarks/scicode/PIPELINE.md)
- [CMPhysBench pipeline](ddsr_bench/benchmarks/cmphysbench/PIPELINE.md)
- [CMT-Benchmark pipeline](ddsr_bench/benchmarks/cmtbench/PIPELINE.md)
- [PHYBench pipeline](ddsr_bench/benchmarks/phybench/PIPELINE.md)
- [PRISM-Physics pipeline](ddsr_bench/benchmarks/prismphysics/PIPELINE.md)
- [PHYSICS pipeline](ddsr_bench/benchmarks/physics/PIPELINE.md)
- [HLE pipeline](ddsr_bench/benchmarks/hle/PIPELINE.md)
- [FrontierScience pipeline](ddsr_bench/benchmarks/frontierscience/PIPELINE.md)
- [OlympiadBench pipeline](ddsr_bench/benchmarks/olympiadbench/PIPELINE.md)
- [HiPhO pipeline](ddsr_bench/benchmarks/hipho/PIPELINE.md)
- [GPQA pipeline](ddsr_bench/benchmarks/gpqa/PIPELINE.md)
- [MMLU pipeline](ddsr_bench/benchmarks/mmlu/PIPELINE.md)
- [MMMU-Pro pipeline](ddsr_bench/benchmarks/mmmupro/PIPELINE.md)
- [SciBench pipeline](ddsr_bench/benchmarks/scibench/PIPELINE.md)
