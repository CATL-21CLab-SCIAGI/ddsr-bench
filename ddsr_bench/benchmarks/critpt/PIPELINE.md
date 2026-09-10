# CritPt pipeline

One CritPt challenge contains a main problem and may contain ordered,
independently indexed subproblems. Preparation compiles each problem into its
own task so Harbor can schedule problems and attempts independently.

```mermaid
flowchart TD
    A["CritPt challenge JSON"] --> B["Load main and indexed subproblems"]
    B --> C["Compile one task per problem"]
    C --> D{"Runner"}
    D -->|"harbor run"| E["Harbor trial"]
    D -->|"ddsr-solve"| F["Static trial"]
    E --> G{"Prompt strategy"}
    F --> G
    G -->|"one-step"| H["One call: statement + code template"]
    G -->|"two-step"| I["Call 1: derivation without template"]
    I --> J["Call 2: official formatting prompt + template"]
    H --> K["Final response"]
    J --> K
    K --> L["Extract first Python block"]
    L --> M["Validate answer function and AST"]
    M -->|"invalid"| N["Failed trial"]
    M -->|"static"| O["Validation result; reward is null"]
    M -->|"Harbor"| P["Upload answer.py"]
    P --> Q["No-network container verifier"]
    Q --> R{"Private reference available?"}
    R -->|"no"| S["Format result"]
    R -->|"yes"| T["Run answer and real_answer"]
    T --> U["Compare outputs for verifier inputs"]
    S --> V["Reward and status"]
    U --> V
    N --> W["Trial directory"]
    O --> W
    V --> W
    W --> X["Collect or export"]
    X --> Y["Optional explicit 70-main submission"]
```

Prepared job runs generate every problem as an independent trial. The lower-level
challenge conversation API additionally supports carrying prior subproblem
history forward or generating subproblems independently. One-step uses one model
call per problem; two-step uses two calls but still produces one artifact and
one trial result.

Official public challenges do not expose reference answers or testcases, so
their local Harbor result is format validation. Reference execution is used
only when verifier-side answer data is available. Artificial Analysis
submission is a separate, explicit action after collecting one complete batch.

See the [CritPt guide](README.md) for commands and data setup.
