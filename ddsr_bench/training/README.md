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

For CMPhysBench, `native` and `full` both export its single recorded call. The
completion contains the visible derivation and boxed answer; provider reasoning
is retained only in the canonical trajectory.

## Sample records

Every JSONL line has one conversational prompt and one supervised assistant
completion. Long prompt text is abbreviated below.

A native CritPt formatting sample preserves the preceding derivation:

```json
{"id":"trial:formatting","prompt":[{"role":"system","content":"..."},{"role":"user","content":"problem"},{"role":"assistant","content":"derivation"},{"role":"user","content":"format with template"}],"completion":[{"role":"assistant","content":"```python\n...\n```"}],"metadata":{"benchmark":"critpt","stage":"formatting"}}
```

The derived CritPt answer view turns the same two-step result into one-step SFT:

```json
{"id":"trial:answer","prompt":[{"role":"system","content":"one-step prompt"},{"role":"user","content":"problem + template"}],"completion":[{"role":"assistant","content":"derivation\n\n```python\n...\n```"}],"metadata":{"benchmark":"critpt","stage":"answer","derived":true}}
```

A SciCode native sample represents one sequential model-generated step:

```json
{"id":"trial:19.2","prompt":[{"role":"user","content":"problem + preceding code + next function"}],"completion":[{"role":"assistant","content":"def step_2(...): ..."}],"metadata":{"benchmark":"scicode","stage":"19.2"}}
```

A CMPhysBench sample preserves the official system and constructed user input:

```json
{"id":"trial:answer","prompt":[{"role":"system","content":"..."},{"role":"user","content":"question + symbols"}],"completion":[{"role":"assistant","content":"derivation + boxed answer"}],"metadata":{"benchmark":"cmphysbench","stage":"answer"}}
```

When a provider exposes hidden reasoning, the canonical trajectory retains it
under a separate `reasoning` key. SFT samples intentionally supervise the
visible `content`; CritPt's derived answer sample uses its visible derivation.
