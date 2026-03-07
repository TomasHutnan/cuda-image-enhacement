from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from tgpu.paths import DATASETS_DIR, MANIFESTS_DIR, get_dataset_root, get_manifest_path, to_repo_relative


DEFAULT_PATTERNS = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.bmp")
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ManifestFile:
    path: str
    size_bytes: int
    sha256: str


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_matching_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(files))


def build_manifest(dataset_name: str, patterns: Iterable[str] = DEFAULT_PATTERNS) -> dict:
    resolved_root = get_dataset_root(dataset_name).resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {resolved_root}")
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {resolved_root}")

    file_records: list[ManifestFile] = []
    total_size_bytes = 0

    for file_path in iter_matching_files(resolved_root, patterns):
        size_bytes = file_path.stat().st_size
        total_size_bytes += size_bytes
        file_records.append(
            ManifestFile(
                path=file_path.relative_to(resolved_root).as_posix(),
                size_bytes=size_bytes,
                sha256=compute_sha256(file_path),
            )
        )

    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": dataset_name,
            "root_rel": to_repo_relative(resolved_root),
            "patterns": list(patterns),
            "file_count": len(file_records),
            "total_size_bytes": total_size_bytes,
        },
        "files": [file_record.__dict__ for file_record in file_records],
    }


def write_manifest(manifest: dict, output_path: Path) -> Path:
    resolved_output = output_path.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return resolved_output


def load_manifest(dataset_name: str) -> dict:
    manifest_path = get_manifest_path(dataset_name).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def list_datasets() -> list[dict[str, object]]:
    dataset_names = {path.name for path in DATASETS_DIR.iterdir() if path.is_dir()} if DATASETS_DIR.exists() else set()
    manifest_names = {path.stem for path in MANIFESTS_DIR.glob("*.json")} if MANIFESTS_DIR.exists() else set()

    items: list[dict[str, object]] = []
    for name in sorted(dataset_names | manifest_names):
        dataset_root = get_dataset_root(name)
        manifest_path = get_manifest_path(name)
        items.append(
            {
                "name": name,
                "dataset_exists": dataset_root.is_dir(),
                "manifest_exists": manifest_path.is_file(),
                "dataset_root": to_repo_relative(dataset_root) if dataset_root.exists() else dataset_root.as_posix(),
                "manifest_path": to_repo_relative(manifest_path) if manifest_path.exists() else manifest_path.as_posix(),
            }
        )
    return items


def inspect_manifest(dataset_name: str) -> dict[str, object]:
    manifest = load_manifest(dataset_name)
    dataset = manifest.get("dataset", {})
    files = manifest.get("files", [])
    extension_counts = Counter()

    for file_record in files:
        suffix = Path(file_record["path"]).suffix.lower() or "<no extension>"
        extension_counts[suffix] += 1

    return {
        "name": dataset.get("name", dataset_name),
        "root_rel": dataset.get("root_rel", ""),
        "file_count": dataset.get("file_count", len(files)),
        "total_size_bytes": dataset.get("total_size_bytes", 0),
        "patterns": dataset.get("patterns", []),
        "created_at": manifest.get("created_at", ""),
        "manifest_version": manifest.get("manifest_version", MANIFEST_VERSION),
        "extension_counts": dict(sorted(extension_counts.items())),
    }
