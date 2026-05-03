# pv162-tescan-gpu

CUDA-focused image processing project for the PV162 TESCAN collaboration.

## Summary

This project delivers a production-oriented CUDA reimplementation of the reference TESCAN grayscale enhancement pipeline. The current system executes a four-stage GPU workflow (Non-local Means, Unsharp Mask, Richardson-Lucy, Histogram Stretch), preserves stable stage-level outputs for validation, and supports raw grayscale inputs (`u8` and `u16`) with normalized float processing on device.

### What Was Delivered

- End-to-end GPU pipeline with stage orchestration and stable stage naming.
- Device-resident processing flow with optional intermediate stage capture.
- Test-backed implementation with Catch2 and stage-focused validation workflows.
- Performance-focused kernel optimizations (including NLM and RL improvements).
- Usable tooling for CLI runs, stage isolation, reference comparison, and dataset manifest validation.
- Stream-viewer and OpenGL experimental visualization support for interactive inspection.

### Quality and Verification Status

- C++ unit test suite is integrated and passing in regular development workflow.
- Python reference wrapper and stage-comparison metrics support parity checks (`MAE`, `RMSE`, `PSNR`, `SSIM`).
- Documentation covers algorithm contract, expanded buffer layout, stage naming conventions, and dataset integrity.

### Recent Technical Update

- Border handling was upgraded from repeated-edge clamping to symmetric mirroring while preserving expanded-buffer architecture.
- The change improves edge behavior quality with low practical runtime impact and no regression in existing tests.

### Current Outcome

The project now provides a robust GPU processing baseline suitable for further parity tuning, benchmark-driven optimization, and course/demo delivery.

## Setup

The C++ build uses `vcpkg` manifest mode.

1. Set the `VCPKG_ROOT` environment variable to your local `vcpkg` installation.
	For CUDA work, this project is expected to use a VS 2022-compatible toolchain.
	Configure your preset in `CMakeUserPresets.json`.
2. Configure the project with your local preset:

```bash
cmake --preset local
```

3. Build it:

```bash
cmake --build --preset local
```

4. Run the C++ executable:

```bash
build\Debug\tgpu_cli.exe <input> <output> --dump-stages <directory>
```

5. Run the C++ unit tests:

```bash
ctest --preset local
```

## Run

Set up the Python environment and use the project CLI:

```bash
python scripts/tgpu.py -h
```

## Docs
- Dataset manifest and integrity workflow is documented in [docs/data-validation.md](docs/data-validation.md).
- Algorithm behavior and stage-by-stage reference is documented in [docs/algorithm-specification.md](docs/algorithm-specification.md).
- Expanded GPU buffer layout and indexing rules are documented in [docs/expanded-buffer-layout.md](docs/expanded-buffer-layout.md).
- Pipeline stage naming and validation guidance is documented in [docs/validation.md](docs/validation.md).