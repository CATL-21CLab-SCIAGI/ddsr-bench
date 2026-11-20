# OlympiadBench pipeline

```mermaid
flowchart TD
    A["Pinned 18-subset official release"] --> B["Public question, context, and images"]
    A --> C["Verifier-only final answers and tolerances"]
    B --> D["Exact language/type-specific official prompt"]
    D --> E["Shared multimodal model client"]
    E --> F["One generated solution"]
    F --> G{"Problem kind"}
    C --> G
    G -->|"Open-ended"| H["Official deterministic symbolic scorer"]
    G -->|"Theorem proof"| I["Explicit ungraded result"]
    H --> J["Pass / different / error"]
    I --> K["Collect result or export trajectory"]
    J --> K
```

One trial is one problem-attempt pair in its own directory. Static evaluation is
the primary path because no generated program is executed. Open-ended scoring is
bounded by the job runner and uses no network. Proof responses remain available
for later human or explicitly configured judge analysis, but do not enter the
official automatic accuracy denominator.
