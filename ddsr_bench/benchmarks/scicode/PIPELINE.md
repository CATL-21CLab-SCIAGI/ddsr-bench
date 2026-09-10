# SciCode pipeline

One SciCode task contains ordered function-writing steps. Later prompts include
the code accumulated from earlier steps, so generation inside a trial is
sequential even when Harbor runs different tasks concurrently.

```mermaid
flowchart TD
    A["SciCode dataset record"] --> B["Load problem, steps, and tests"]
    B --> C["Compile one task per complete problem"]
    C --> D["Harbor schedules problem × attempt"]
    D --> E["Decode public problem instruction"]
    E --> F["Start with provided dependencies"]
    F --> G{"Next ordered step"}
    G -->|"Pinned compatibility fix"| H["Use bundled fixed code"]
    G -->|"Model step"| I["Prompt with preceding functions"]
    I --> J["Generate and extract step code"]
    H --> K["Append function to solution"]
    J --> K
    K -->|"More steps"| G
    K -->|"Complete"| L["Upload solution.py"]
    L --> M["No-network container verifier"]
    M --> N["Run each evaluated step's assertions"]
    N --> O{"Every tested step passes?"}
    O -->|"yes"| P["Reward 1"]
    O -->|"no"| Q["Reward 0"]
    P --> R["Trial directory"]
    Q --> R
    R --> S["Collect or export"]
```

The verifier executes each step in a separate isolated Python subprocess using
the official assertions and targets from `test_data.h5`. The complete problem
passes only when every evaluated step passes. SciCode has no static runner and
no Artificial Analysis submission stage.

Trajectory export emits one training sample per model-generated step. Bundled
fixed steps remain context, not training targets.

See the [SciCode guide](README.md) for commands and data setup.
