# Agent guidance

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for repository structure, conventions,
and checks. Before adding or renaming an API, inspect the closest existing
benchmark adapter and preserve its style. Keep handwritten increments near 200
lines when practical, stop for review between increments, and never commit
generated tasks, outputs, caches, secrets, or empty directories.

## Maintaining guidance

When an agreed project rule changes, update its concise contributor-facing form
here or in CONTRIBUTING.md and align the personal `ddsr-bench-maintenance` skill
when available. Keep detailed rationale in the skill, not duplicated wholesale
here. Repository guidance must stand alone without personal files. Remove
superseded rules; report any skill update that cannot be completed.

## Documentation changes

After moving, removing, or renaming sections, check incoming links and prose
references across tracked Markdown. Validate section anchors as well as file
paths, including headings changed by numbering or emojis. Code-layout references
belong in [CONTRIBUTING.md](CONTRIBUTING.md#code-layout), not the root README.
Fix stale references rather than restoring intentionally removed sections.

## Shared-helper review

Before adding helpers, search for equivalent operations across the package,
including inline code and differently named functions. Audit related readers,
writers, credential lookup, and validation together. Preserve atomic writes,
exclusive-create rules, streaming JSONL, strict consensus checks, and Harbor's
credential precedence. Prefer existing utility modules; do not deduplicate
benchmark-specific or vendored scoring behavior merely because it looks alike.
Test failure semantics, remove obsolete helpers, and report deliberate duplicates
or deferred work explicitly.

## Benchmark integration review

When adding or auditing a benchmark:

- Pin repository commits, dataset revisions, prompts, scoring code, licenses,
  and published inference settings in `UPSTREAM.md`. Distinguish upstream
  behavior from project choices where roles, extraction, or API details are
  unpublished.
- Exercise real pinned data when available. Check field shapes, unique IDs,
  gradable counts, public/private projection, and prepared-task fallback using
  the exceptions actually raised by the data library.
- Reuse one generation adapter across static and Harbor paths. Keep solutions,
  references, and verifier state out of model messages, logs, and trajectories.
- Compare rendered prompts and adapted scorers with upstream behavior. Test
  malformed output, exact metric boundaries, and potentially hanging scorers
  behind a process boundary.
- Run a real static trial and, when supported, one real Harbor trial. DSW or
  remote GPU checks are needed only for deployment-specific behavior.
- Build and inspect a wheel for prompts, configs, assets, and vendored license
  notices. Run Black, Ruff, all tests, and `git diff --check`; verify links,
  ignored artifacts, empty directories, and final `git status`.
- Update the benchmark guide and pipeline plus the root catalog and pipeline.
  Report mocked and real checks separately and disclose remaining limitations.
