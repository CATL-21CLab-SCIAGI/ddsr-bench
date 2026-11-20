# SciBench pipeline

```mermaid
flowchart TD
    A["Pinned official repository data"] --> B["580 text + 151 image targets"]
    A --> C["Released solved examples"]
    B --> D{"Prompt strategy"}
    C -->|"few / few-cot text only"| D
    D --> E["Shared chat client"]
    E -->|"zero-cot only"| F["Derivation embedded in fresh answer request"]
    E -->|"other strategies"| G["Final response"]
    F --> G
    G --> H["Last boxed-value extraction"]
    H --> I["Scientific-unit rescaling"]
    I --> J["5% relative-tolerance score"]
    J --> K["Split and textbook summaries"]
    G --> L["Reference-free trajectory export"]
```

One trial is one target/strategy attempt. Image targets are official released
data, but their chat-message envelope is a documented DDSR choice because the
pinned repository publishes no multimodal inference script.
