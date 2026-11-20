# HiPhO pipeline

```mermaid
flowchart TD
    A["Pinned 14-exam official release"] --> B["Public question, data sheet, and images"]
    A --> C["Verifier-only answers, points, and marking scheme"]
    B --> D["Exact English or Chinese VLMEvalKit prompt"]
    D --> E["Shared multimodal model client"]
    E --> F["One worked solution"]
    F --> G["Official answer-level verifier"]
    F --> H["Gemini step-level rubric judge"]
    C --> G
    C --> H
    G --> I["Maximum of answer and step points"]
    H --> I
    I --> J["Per-exam totals and attempt averages"]
    F --> K["Reference-free trajectory export"]
```

One trial is one subquestion-attempt pair with an independent directory. The
published protocol uses eight attempts, averages each subquestion across those
attempts, sums within each exam, and reports the mean of 13 exam totals. DDSR
also exposes the 14-exam current release and always labels the chosen cohort.

The scorer runs on the host because it parses text and calls a judge; it never
executes model-generated code. Prepared tasks provide the common offline data
layout and Hugging Face fallback, not a reason to invoke Harbor.
