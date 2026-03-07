# Validation

## Stage Naming

Intermediate stage dumps use stable stage identifiers and a sortable numeric prefix:

```text
00_input_normalized.png
10_non_local_means.png
20_unsharp_mask.png
30_richardson_lucy.png
40_histogram_stretch.png
90_output.png
```

- The stage `name` is the real identifier and must stay stable across implementations.
- The numeric prefix is only for sort order in file listings.
- Prefixes should leave gaps, usually by tens, so new stages can be inserted later without renaming every file.
- Python and C++ dumps should use the same stage names.

## Test Split

- C++ unit tests for local logic, small deterministic helpers, parameter validation, border handling, shape checks, and simple stage-specific invariants.
- Python-driven integration tests for end-to-end and stage-by-stage comparison against the reference pipeline.
