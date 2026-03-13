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

## Pytest Integration Tests (TBD)

- How to compare outputs with reference?
- How to find suitable thresholds for comparisson?

## Reference Stage Capture Wrapper

```bash
python scripts/tgpu.py reference capture-stages <input_image> <output_dir> --save-bit-depth u16
```

This writes the same stage-name sequence used by C++:

```text
00_input_normalized.png
10_non_local_means.png
20_unsharp_mask.png
30_richardson_lucy.png
40_histogram_stretch.png
90_output.png
```

## Stage Comparison Command

Compare two stage-capture directories and report per-stage `MAE`, `RMSE`, `PSNR`, and `SSIM`:

```bash
python scripts/tgpu.py reference compare-stages <reference_dir> <candidate_dir>
```

To also compute border-free metrics, crop an equal margin from all sides before scoring:

```bash
python scripts/tgpu.py reference compare-stages <reference_dir> <candidate_dir> --crop-border-px 16
```

Expected convention for both directories:

```text
00_input_normalized.png
10_non_local_means.png
20_unsharp_mask.png
30_richardson_lucy.png
40_histogram_stretch.png
90_output.png
```

Metrics are computed on normalized `[0, 1]` images.

The command also reports stage effect metrics (`Δ`) that compare each stage with the previous stage inside each pipeline.

This helps identify placeholder stages: when reference `Δ` is significant but C++ `Δ` is near zero, the C++ stage is likely still close to passthrough.