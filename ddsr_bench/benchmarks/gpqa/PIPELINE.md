# GPQA pipeline

```mermaid
flowchart TD
    A["Pinned password-protected official archive"] --> B["Main, Diamond, or Extended cohort"]
    B --> C["Seeded answer-choice shuffle"]
    C --> D["Official zero-shot prompt"]
    D --> E["Shared model client"]
    E --> F["One text response"]
    F --> G["Official A-D response parser"]
    B --> H["Verifier-only answer letter"]
    G --> I["Exact-match accuracy"]
    H --> I
    I --> J["Overall and grouped summaries"]
    F --> K["Reference-free trajectory export"]
```

One trial is one question-attempt pair with an independent output directory.
The verifier parses a single A-D choice from the response and assigns zero to
malformed output. It does not use the released expert explanation for scoring.

Static evaluation is protocol-complete because GPQA requires no code execution
or environment state. Prepared tasks provide an offline copy of the default
Main cohort; named alternate cohorts are loaded from the pinned raw archive.
