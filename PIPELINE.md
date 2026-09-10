# Evaluation pipeline

`ddsr-solve` stops after code extraction and static AST validation. Harbor adds
container isolation and, when verifier data exists, answer execution and comparison.

| Mode | Entry point | Code execution | Result |
| --- | --- | --- | --- |
| Harbor | [`CritPtAgent.run`](ddsr_bench/benchmarks/critpt/harbor.py) | When a reference or testcases exist | Reward and status |
| Static | [`run_trial`](ddsr_bench/benchmarks/critpt/static.py) | None | Validation status; reward is `null` |

Both modes use the same prepared tasks, model clients, prompts, generation
runner, answer extraction, and AST validation. Only Harbor continues into the
container verifier.

```mermaid
flowchart TD
    A["CritPt challenge JSON"] --> B["Load main and indexed subproblems"]
    B --> C["Compile each problem into a Harbor task"]
    C --> D{"Runner"}
    D -->|"harbor run"| E["Harbor schedules problem × attempt"]
    D -->|"ddsr-solve"| ES["Static runner schedules problem × attempt"]
    E --> F["One trial receives public ProblemSpec"]
    ES --> F

    F --> G{"Prompt strategy"}
    G -->|"one-step"| H["One LLM call<br/>problem + template"]
    G -->|"two-step"| I["LLM call 1<br/>reasoning without template"]
    I --> J["LLM call 2<br/>formatting prompt + template"]
    H --> K["Final model response"]
    J --> K

    K --> L["Extract code and validate AST"]
    L -->|"invalid"| M["Failed trial<br/>no answer artifact"]
    L -->|"valid + Harbor"| N["Upload /app/answer.py"]
    L -->|"valid + static"| NS["Validated; reward is null"]
    N --> O["No-network verifier"]
    O --> OA{"Reference or testcases?"}
    OA -->|"yes"| OB["Execute and compare answer"]
    OA -->|"no"| OC["Format validation only"]
    OB --> P["Reward and status"]
    OC --> P

    M --> Q["Harbor or static trial directory"]
    NS --> Q
    P --> Q
    Q --> R["Collect all job trials"]
    R --> S["Group identical agent, model, and strategy"]
    S --> T["Assign per-problem attempt indices"]
    T --> U{"One trial for every problem?"}
    U -->|"no"| V["Report unbatched trials"]
    U -->|"yes"| W["Complete attempt batch"]

    W --> X["Write summary.json and summary.csv"]
    X --> Y["User selects one attempt"]
    Y --> Z{"Exactly 70 official main IDs<br/>with answer artifacts?"}
    Z -->|"no"| ZA["Reject submission"]
    Z -->|"yes"| ZB["Submit once to Artificial Analysis"]
    ZB --> ZC["Save submission response"]
```

A trial is one complete evaluation of one problem for one attempt. A two-step
trial contains two model calls but still produces one answer and one trial
result. Concurrency changes how many trials run simultaneously, not how trials
are grouped into submission attempts.
