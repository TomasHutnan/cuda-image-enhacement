from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MANIFESTS_DIR = DATA_DIR / "manifests"


def get_dataset_root(dataset_name: str) -> Path:
    return DATASETS_DIR / dataset_name


def get_manifest_path(dataset_name: str) -> Path:
    return MANIFESTS_DIR / f"{dataset_name}.json"


def to_repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()
