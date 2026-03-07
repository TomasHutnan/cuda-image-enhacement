# Data Validation

This project keeps large datasets outside Git history but inside a stable repo-local layout:

```text
data/
  datasets/
    shift/
  manifests/
    shift.json
```

- Put raw datasets in `data/datasets/<name>`.
- Commit manifests in `data/manifests/<name>.json`.
- Keep `data/datasets` ignored in Git.

## Manifest CLI

The dataset integrity CLI lives at `scripts/tgpu.py`.

Build or rebuild a manifest for a dataset:

```bash
python scripts/tgpu.py manifest build shift
```

Verify the dataset contents against its manifest:

```bash
python scripts/tgpu.py manifest verify shift
```

List known datasets and whether a manifest exists:

```bash
python scripts/tgpu.py manifest list
```

Inspect manifest metadata and extension counts:

```bash
python scripts/tgpu.py manifest inspect shift
```

## Notes

- Manifests store repo-relative dataset roots, not absolute machine paths.
- Dataset names are the main interface. `shift` maps to `data/datasets/shift` and `data/manifests/shift.json`.
- If you move a dataset, rebuild its manifest so `root_rel` stays correct.