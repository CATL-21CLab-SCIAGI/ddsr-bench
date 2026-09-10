# Benchmarks

Each benchmark owns its source-data setup, generation contract, verifier, and
limitations. Shared clients, collection, and export use the same interfaces
across all benchmarks.

| Benchmark | Task | Evaluation | Documentation |
| --- | --- | --- | --- |
| CritPt | Physics derivation returned as executable answers | Static validation or Harbor; optional official submission | [Guide](ddsr_bench/benchmarks/critpt/README.md) · [Pipeline](ddsr_bench/benchmarks/critpt/PIPELINE.md) |
| SciCode | Sequential scientific function generation | Harbor execution of official assertions | [Guide](ddsr_bench/benchmarks/scicode/README.md) · [Pipeline](ddsr_bench/benchmarks/scicode/PIPELINE.md) |

New integrations should follow the capability layout in the root
[README](README.md#repository-layout) and provide both a guide and a pipeline.
