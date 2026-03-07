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

## Catch2 Unit Tests

- Build the test target with the normal CMake build.
- Run all C++ tests with `ctest --preset default`.