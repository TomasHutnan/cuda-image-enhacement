# pv162-tescan-gpu

CUDA-focused image processing project for the PV162 TESCAN collaboration.

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