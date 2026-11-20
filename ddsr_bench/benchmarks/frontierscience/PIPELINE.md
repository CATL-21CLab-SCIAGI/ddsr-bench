# FrontierScience pipeline

```mermaid
flowchart TD
    A["Pinned 160-record gold set"] --> B["Public problem prompt"]
    A --> C["Verifier-only answer or rubric"]
    B --> D["Shared model client"]
    D --> E["One scientific response"]
    E --> F{"Track"}
    C --> F
    F -->|"Olympiad"| G["Official equivalence judge prompt"]
    F -->|"Research"| H["Official ten-point rubric prompt"]
    G --> I["Binary correctness"]
    H --> J["Normalized rubric score + 7/10 success"]
    I --> K["Collect result or export trajectory"]
    J --> K
```

One trial is one problem-attempt pair followed by one judge request. Empty model
answers score zero without invoking the judge. A judge transport or parse failure
is recorded as unjudged rather than silently repaired. Separate trial directories
prevent concurrent attempts from overwriting one another.
