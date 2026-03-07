# pv162-tescan-gpu

CUDA-focused image processing project for the PV162 TESCAN collaboration.

## Setup

The C++ build uses `vcpkg` manifest mode.

1. Set the `VCPKG_ROOT` environment variable to your local `vcpkg` installation.
2. Configure the project with the default preset:

```bash
cmake --preset default
```

3. Build it:

```bash
cmake --build --preset default
```

4. Run the C++ executable:

```bash
build\Debug\tgpu_cli.exe <input> <output> --dump-stages <directory>
```

## Run

Set up the Python environment and use the project CLI:

```bash
python scripts/tgpu.py -h
```

Dataset manifest and integrity workflow is documented in [docs/data-validation.md](docs/data-validation.md).