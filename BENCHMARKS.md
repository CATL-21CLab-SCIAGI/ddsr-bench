# Benchmarks

Each benchmark owns its source-data setup, generation contract, verifier, and
limitations. Shared clients, collection, and export use the same interfaces
across all benchmarks.

| Benchmark | Task | Evaluation | Documentation |
| --- | --- | --- | --- |
| CritPt | Physics derivation returned as executable answers | Static validation or Harbor; optional official submission | [Guide](ddsr_bench/benchmarks/critpt/README.md) · [Pipeline](ddsr_bench/benchmarks/critpt/PIPELINE.md) |
| SciCode | Sequential scientific function generation | Harbor execution of official assertions | [Guide](ddsr_bench/benchmarks/scicode/README.md) · [Pipeline](ddsr_bench/benchmarks/scicode/PIPELINE.md) |
| CMPhysBench | Condensed-matter reasoning with boxed LaTeX answers | Direct static SEED scoring; optional Harbor | [Guide](ddsr_bench/benchmarks/cmphysbench/README.md) · [Pipeline](ddsr_bench/benchmarks/cmphysbench/PIPELINE.md) |
| CMT-Benchmark | Condensed-matter formulas with symbolic equivalence | Direct static binary scoring; optional Harbor | [Guide](ddsr_bench/benchmarks/cmtbench/README.md) · [Pipeline](ddsr_bench/benchmarks/cmtbench/PIPELINE.md) |
| PHYBench | Advanced physics reasoning with boxed LaTeX answers | Direct static EED scoring; optional Harbor | [Guide](ddsr_bench/benchmarks/phybench/README.md) · [Pipeline](ddsr_bench/benchmarks/phybench/PIPELINE.md) |
| PRISM-Physics | Process-level physics reasoning over formula DAGs | Direct static ancestor-closure scoring; optional Harbor | [Guide](ddsr_bench/benchmarks/prismphysics/README.md) · [Pipeline](ddsr_bench/benchmarks/prismphysics/PIPELINE.md) |
| PHYSICS | University-level text and multimodal physics problems | Direct static answer-level LLM judging | [Guide](ddsr_bench/benchmarks/physics/README.md) · [Pipeline](ddsr_bench/benchmarks/physics/PIPELINE.md) |
| HLE-Verified | Multidisciplinary closed-ended text and image questions | Direct static official-style LLM judging | [Guide](ddsr_bench/benchmarks/hle/README.md) · [Pipeline](ddsr_bench/benchmarks/hle/PIPELINE.md) |
| FrontierScience | Expert scientific reasoning across two evaluation tracks | Direct static official LLM judging | [Guide](ddsr_bench/benchmarks/frontierscience/README.md) · [Pipeline](ddsr_bench/benchmarks/frontierscience/PIPELINE.md) |
| OlympiadBench | Bilingual multimodal mathematics and physics competitions | Direct static official symbolic scoring for open-ended tasks; proofs retained ungraded | [Guide](ddsr_bench/benchmarks/olympiadbench/README.md) · [Pipeline](ddsr_bench/benchmarks/olympiadbench/PIPELINE.md) |
| HiPhO | Multimodal high-school physics Olympiads with human marking schemes | Direct static official answer- and step-level grading | [Guide](ddsr_bench/benchmarks/hipho/README.md) · [Pipeline](ddsr_bench/benchmarks/hipho/PIPELINE.md) |
| GPQA | Graduate-level biology, chemistry, and physics multiple choice | Direct static official sampled-answer scoring | [Guide](ddsr_bench/benchmarks/gpqa/README.md) · [Pipeline](ddsr_bench/benchmarks/gpqa/PIPELINE.md) |
| MMLU | Knowledge and reasoning across 57 academic and professional subjects | Direct static five-shot generated-letter scoring | [Guide](ddsr_bench/benchmarks/mmlu/README.md) · [Pipeline](ddsr_bench/benchmarks/mmlu/PIPELINE.md) |
| MMMU-Pro | Multidisciplinary multimodal reasoning across Standard and Vision settings | Direct static official answer-letter scoring | [Guide](ddsr_bench/benchmarks/mmmupro/README.md) · [Pipeline](ddsr_bench/benchmarks/mmmupro/PIPELINE.md) |
| SciBench | College-level text and image scientific problem solving | Direct static official numeric scoring | [Guide](ddsr_bench/benchmarks/scibench/README.md) · [Pipeline](ddsr_bench/benchmarks/scibench/PIPELINE.md) |

New integrations should follow the [code layout](CONTRIBUTING.md#code-layout)
and provide both a guide and a pipeline.

The selected source edition and accessible correction alternatives are recorded
in the [dataset catalog](DATASETS.md).
