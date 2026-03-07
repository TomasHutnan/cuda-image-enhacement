from __future__ import annotations

import argparse
from typing import cast

from tgpu.manifest import DEFAULT_PATTERNS, build_manifest, inspect_manifest, list_datasets, write_manifest
from tgpu.paths import get_manifest_path
from tgpu.verify import verify_manifest


def add_manifest_subcommands(subparsers: argparse._SubParsersAction) -> None:
    manifest_parser = subparsers.add_parser("manifest", help="Build and verify dataset manifests")
    manifest_subparsers = manifest_parser.add_subparsers(dest="manifest_command", required=True)

    build_parser = manifest_subparsers.add_parser("build", help="Create a manifest for a named dataset")
    build_parser.add_argument("name", help="Dataset name under data/datasets/<name>")
    build_parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help="Glob pattern to include. May be provided multiple times.",
    )

    verify_parser = manifest_subparsers.add_parser("verify", help="Verify a named dataset against its manifest")
    verify_parser.add_argument("name", help="Dataset name under data/datasets/<name>")

    manifest_subparsers.add_parser("list", help="List dataset directories and matching manifests")

    inspect_parser = manifest_subparsers.add_parser("inspect", help="Show manifest summary for a named dataset")
    inspect_parser.add_argument("name", help="Dataset name under data/datasets/<name>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tgpu", description="Project data management utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_manifest_subcommands(subparsers)
    return parser.parse_args()


def handle_manifest_build(args: argparse.Namespace) -> int:
    patterns = tuple(args.patterns or DEFAULT_PATTERNS)
    manifest = build_manifest(args.name, patterns=patterns)
    output_path = get_manifest_path(args.name)
    written_manifest = write_manifest(manifest, output_path)
    print(f"Wrote manifest with {manifest['dataset']['file_count']} files to {written_manifest}")
    return 0


def handle_manifest_verify(args: argparse.Namespace) -> int:
    issues = verify_manifest(args.name)
    if not issues:
        print("Manifest verification succeeded")
        return 0

    print(f"Manifest verification failed with {len(issues)} issue(s):")
    for issue in issues[:20]:
        print(f"- {issue.path}: {issue.problem}")
    if len(issues) > 20:
        print(f"- ... {len(issues) - 20} more issue(s)")
    return 1


def handle_manifest_list() -> int:
    items = list_datasets()
    if not items:
        print("No datasets or manifests found")
        return 0

    for item in items:
        status = []
        status.append("dataset" if item["dataset_exists"] else "no-dataset")
        status.append("manifest" if item["manifest_exists"] else "no-manifest")
        print(f"{item['name']}: {', '.join(status)}")
        print(f"  dataset: {item['dataset_root']}")
        print(f"  manifest: {item['manifest_path']}")
    return 0


def handle_manifest_inspect(args: argparse.Namespace) -> int:
    summary = inspect_manifest(args.name)
    patterns = cast(list[str], summary["patterns"])
    extension_counts = cast(dict[str, int], summary["extension_counts"])
    print(f"name: {summary['name']}")
    print(f"root: {summary['root_rel']}")
    print(f"manifest version: {summary['manifest_version']}")
    print(f"created at: {summary['created_at']}")
    print(f"file count: {summary['file_count']}")
    print(f"total size bytes: {summary['total_size_bytes']}")
    print(f"patterns: {', '.join(patterns)}")
    print("extensions:")
    for extension, count in extension_counts.items():
        print(f"  {extension}: {count}")
    return 0


def main() -> int:
    args = parse_args()

    if args.command == "manifest":
        if args.manifest_command == "build":
            return handle_manifest_build(args)
        if args.manifest_command == "verify":
            return handle_manifest_verify(args)
        if args.manifest_command == "list":
            return handle_manifest_list()
        if args.manifest_command == "inspect":
            return handle_manifest_inspect(args)

    raise ValueError(f"Unsupported command combination: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
