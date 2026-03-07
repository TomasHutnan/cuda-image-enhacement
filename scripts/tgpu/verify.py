from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tgpu.manifest import compute_sha256
from tgpu.paths import REPO_ROOT, get_manifest_path


@dataclass(frozen=True)
class VerificationIssue:
    path: str
    problem: str


def load_manifest(manifest_path: Path) -> dict:
    resolved_manifest = manifest_path.resolve()
    if not resolved_manifest.exists():
        raise FileNotFoundError(f"Manifest does not exist: {resolved_manifest}")
    return json.loads(resolved_manifest.read_text(encoding="utf-8"))


def verify_manifest(dataset_name: str) -> list[VerificationIssue]:
    manifest = load_manifest(get_manifest_path(dataset_name))
    dataset = manifest.get("dataset", {})
    manifest_root = (REPO_ROOT / dataset.get("root_rel", "")).resolve()
    if not manifest_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {manifest_root}")

    issues: list[VerificationIssue] = []
    expected_paths: set[str] = set()

    for file_record in manifest.get("files", []):
        relative_path = file_record["path"]
        expected_paths.add(relative_path)
        candidate = manifest_root / Path(relative_path)

        if not candidate.exists():
            issues.append(VerificationIssue(path=relative_path, problem="missing"))
            continue

        actual_size = candidate.stat().st_size
        if actual_size != file_record["size_bytes"]:
            issues.append(
                VerificationIssue(
                    path=relative_path,
                    problem=f"size mismatch: expected {file_record['size_bytes']}, got {actual_size}",
                )
            )
            continue

        actual_sha256 = compute_sha256(candidate)
        if actual_sha256 != file_record["sha256"]:
            issues.append(
                VerificationIssue(
                    path=relative_path,
                    problem=f"sha256 mismatch: expected {file_record['sha256']}, got {actual_sha256}",
                )
            )

    for extra_file in sorted(path for path in manifest_root.rglob("*") if path.is_file()):
        relative_path = extra_file.relative_to(manifest_root).as_posix()
        if relative_path not in expected_paths:
            issues.append(VerificationIssue(path=relative_path, problem="unexpected file"))

    return issues
