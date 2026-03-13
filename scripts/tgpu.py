from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from tgpu.manifest import DEFAULT_PATTERNS, build_manifest, inspect_manifest, list_datasets, write_manifest
from tgpu.paths import get_manifest_path
from tgpu.verify import verify_manifest


STAGE_CHOICES = ("non_local_means", "unsharp_mask", "richardson_lucy", "histogram_stretch")


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


def add_reference_subcommands(subparsers: argparse._SubParsersAction) -> None:
    reference_parser = subparsers.add_parser("reference", help="Run reference Python pipeline helpers")
    reference_subparsers = reference_parser.add_subparsers(dest="reference_command", required=True)

    capture_parser = reference_subparsers.add_parser(
        "capture-stages", help="Capture reference pipeline stage outputs with stable stage names"
    )
    capture_parser.add_argument("input", help="Input grayscale image path")
    capture_parser.add_argument("output_dir", help="Directory where captured stage images will be written")
    capture_parser.add_argument(
        "--save-bit-depth",
        choices=("u8", "u16"),
        default="u16",
        help="Bit depth used when writing stage images",
    )
    capture_parser.add_argument(
        "--rl-output-dtype",
        choices=("uint8", "uint16"),
        default="uint16",
        help="Output dtype for the reference Richardson-Lucy stage",
    )
    capture_parser.add_argument(
        "--histogram-sat-percent",
        type=float,
        default=0.5,
        help="Saturation percentile used by the reference histogram stretch",
    )
    capture_parser.add_argument(
        "--only-stage",
        choices=STAGE_CHOICES,
        default=None,
        help="Run only this stage in isolation (all other stages are bypassed)",
    )

    compare_parser = reference_subparsers.add_parser(
        "compare-stages", help="Compare two stage-capture directories and report per-stage metrics"
    )
    compare_parser.add_argument("reference_dir", help="Directory with reference stage captures")
    compare_parser.add_argument("candidate_dir", help="Directory with candidate stage captures")
    compare_parser.add_argument(
        "--crop-border-px",
        type=int,
        default=0,
        help="Ignore this many pixels on each image border before computing border-free metrics",
    )
    compare_parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default=None,
        help="Compare only this stage (useful for isolated-stage validation)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tgpu", description="Project data management utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_manifest_subcommands(subparsers)
    add_reference_subcommands(subparsers)
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


def handle_reference_capture_stages(args: argparse.Namespace) -> int:
    from tgpu.reference_pipeline import capture_reference_stages_from_file

    written_paths = capture_reference_stages_from_file(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        save_bit_depth=args.save_bit_depth,
        rl_output_dtype=args.rl_output_dtype,
        histogram_sat_percent=args.histogram_sat_percent,
        only_stage=args.only_stage,
    )

    print(f"Wrote {len(written_paths)} stage images:")
    for path in written_paths:
        print(f"- {path}")
    return 0


def handle_reference_compare_stages(args: argparse.Namespace) -> int:
    from tgpu.compare import (
        compare_stage_deltas,
        compare_stage_directories,
        format_metrics_table,
        format_stage_delta_table,
    )

    if args.crop_border_px < 0:
        raise ValueError("--crop-border-px must be non-negative")

    reference_path = Path(args.reference_dir)
    candidate_path = Path(args.candidate_dir)
    reference_label = reference_path.name or str(reference_path)
    candidate_label = candidate_path.name or str(candidate_path)

    full_metrics = compare_stage_directories(
        reference_dir=reference_path,
        candidate_dir=candidate_path,
        only_stage=args.stage,
    )
    print("Full image metrics")
    print(format_metrics_table(full_metrics))

    if args.crop_border_px > 0:
        cropped_metrics = compare_stage_directories(
            reference_dir=reference_path,
            candidate_dir=candidate_path,
            crop_border_px=args.crop_border_px,
            only_stage=args.stage,
        )
        print("")
        print(f"Border-free metrics (crop_border_px={args.crop_border_px})")
        print(format_metrics_table(cropped_metrics))

        cropped_stage_deltas = compare_stage_deltas(
            reference_dir=reference_path,
            candidate_dir=candidate_path,
            crop_border_px=args.crop_border_px,
            only_stage=args.stage,
        )
        print("")
        print(f"Border-free stage effect metrics (crop_border_px={args.crop_border_px})")
        print(
            format_stage_delta_table(
                cropped_stage_deltas,
                reference_label=reference_label,
                candidate_label=candidate_label,
            )
        )
    else:
        full_stage_deltas = compare_stage_deltas(
            reference_dir=reference_path,
            candidate_dir=candidate_path,
            crop_border_px=0,
            only_stage=args.stage,
        )
        print("")
        print("Stage effect metrics")
        print(
            format_stage_delta_table(
                full_stage_deltas,
                reference_label=reference_label,
                candidate_label=candidate_label,
            )
        )

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

    if args.command == "reference":
        if args.reference_command == "capture-stages":
            return handle_reference_capture_stages(args)
        if args.reference_command == "compare-stages":
            return handle_reference_compare_stages(args)

    raise ValueError(f"Unsupported command combination: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
