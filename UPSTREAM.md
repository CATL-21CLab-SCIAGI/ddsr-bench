# Upstream provenance

## CritPt

- Repository: https://github.com/CritPt-Benchmark/CritPt
- Commit: `17c2545c302762d2f2d644d923ea4c301605cb08`
- Reused: the prompt text and one-step/two-step conversation behavior.

## Auto_CritPt_Grader

- Repository: https://github.com/CATL-21CLab-SCIAGI/Auto_CritPt_Grader
- Commit: `1787dbb2258af9c606ff97ad6094af5a25e8f246`
- Reused: the deterministic grading contract, with no LLM judge.

## Harbor

- Repository: https://github.com/harbor-framework/harbor
- Pinned package version: `0.22.0`

## SciCode

- Repository: https://github.com/scicode-bench/SciCode
- Commit: `e3158ea011d4235245a547460d3688d7ccbf9900`
- Reimplemented locally: prompts, sequential generation, dataset loading, HDF5
  target decoding, exceptional-step substitutions, and official test execution.
