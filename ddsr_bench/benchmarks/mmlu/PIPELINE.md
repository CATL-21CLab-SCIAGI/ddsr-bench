# MMLU pipeline

```mermaid
flowchart TD
    A["Pinned official 57-subject archive"] --> B["Five dev demonstrations per subject"]
    A --> C["14,042 verifier-only test labels"]
    B --> D["Exact original five-shot prompt"]
    D --> E["Shared chat model client"]
    E --> F["Generated A-D response"]
    F --> G["Strict answer-letter parser"]
    C --> H["Exact-match accuracy"]
    G --> H
    H --> I["Overall, category, and subject summaries"]
    F --> J["Reference-free trajectory export"]
```

One trial is one test-question attempt. DDSR preserves the original prompt but
uses generated answer letters instead of legacy completion-token likelihoods;
the recorded `mmlu-generative` mode makes this protocol distinction explicit.
Static evaluation is complete for this path, and Harbor adds no isolation value.
