# MMMU-Pro pipeline

```mermaid
flowchart TD
    A["Pinned corrected Parquet shards"] --> B{"Setting"}
    B -->|"Standard-10 / Standard-4"| C["Question + labeled choices + placeholder-ordered images"]
    B -->|"Vision"| D["Question screenshot + Vision instruction"]
    C --> E["Official direct or chain-of-thought prompt"]
    D --> E
    E --> F["Shared multimodal chat client"]
    F --> G["Generated response"]
    G --> H["Official answer-letter parser"]
    H --> I["Exact-match accuracy"]
    I --> J["Setting, domain, and subject summaries"]
    G --> K["Reference-free trajectory export"]
```

One trial is one setting-specific question attempt. The official headline is
the arithmetic mean of complete Standard-10 and Vision accuracies. Standard-4
is reported separately. Static evaluation fully implements the released
multiple-choice verifier.
