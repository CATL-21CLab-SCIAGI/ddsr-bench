# Contributing

## Development setup

Use Python 3.12 or newer. From the repository root:

```bash
python -m pip install -e '.[dev]'
```

Install benchmark extras only when needed, for example
`python -m pip install -e '.[dev,cmphysbench]'`.

## Code layout

- `ddsr_bench/benchmarks/`: benchmark-specific adapters and documentation.
- `ddsr_bench/generation/`: shared model clients.
- `ddsr_bench/training/`: shared trajectory and SFT export.
- `configs/`: benchmark, job, prompt, Harbor, and serving configuration.
- `tests/`: shared and benchmark-specific contract checks.

Each benchmark owns its data, generation, evaluation, result, and trajectory
behavior. Shared code coordinates those capabilities but must not hide
benchmark-specific semantics.

## Adding a benchmark

Follow the closest existing adapter and include:

- public/private data projection and pinned upstream provenance;
- exact prompt and evaluation behavior with parity tests;
- registry capabilities and client job configurations;
- benchmark `README.md` and `PIPELINE.md` files;
- result collection and teacher-data export where applicable.

Update the root benchmark catalog and framework pipeline without dropping links
to existing benchmarks. Keep copied upstream code attributed under its license.
The reusable integration and release checklist is recorded in
[AGENTS.md](AGENTS.md#benchmark-integration-review).

## Coding conventions

Use Black's default formatting and the repository's existing naming style. Keep
functions small, use frozen slotted dataclasses for normalized records, and
reserve `Any` for external boundaries. Raise `TypeError` for wrong input types
and `ValueError` for invalid values. Test observable behavior through public
entry points, including malformed input and private-data leakage.

## Checks and pull requests

Run before requesting review:

```bash
python -m black --check .
python -m ruff check .
python -m pytest -q
```

Work on a focused branch and open a pull request into protected `main`. Describe
the behavior changed, validation performed, provenance, and known limitations.
Do not include generated tasks, run outputs, model artifacts, credentials,
caches, or empty directories.
