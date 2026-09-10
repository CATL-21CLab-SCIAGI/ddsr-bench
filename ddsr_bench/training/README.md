# Training data

Trajectory export first converts each benchmark's logs into the same canonical
schema. One `Generation` represents one request that was sent to a model and its
response. Provider reasoning is retained separately from visible response
content.

Two views apply to every benchmark:

| View | Samples |
| --- | --- |
| `native` | Only model calls recorded during inference |
| `full` | Recorded calls plus benchmark-specific derived samples |

Each benchmark adapter decides how those views map to its trajectory and may
provide additional views. CritPt provides three:

| CritPt view | Sample |
| --- | --- |
| `derivation` | First call of a two-step run |
| `formatting` | Second call of a two-step run |
| `answer` | One-step answer representation |

For a two-step CritPt trajectory, inference records these calls:

```text
problem -> derivation
problem + derivation + formatting request -> code
```

The `native` view exports exactly those two prompt/completion pairs. The `full`
view also derives a third training sample:

```text
problem + code template -> derivation + code
```

That third sample was not a separate inference call; it is an intentional
one-step augmentation. For SciCode, `native` and `full` currently both export
one sample per model-generated step. Fixed compatibility steps remain in the
trajectory for provenance and context but are never SFT targets.
