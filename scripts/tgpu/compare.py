from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from tgpu.reference_pipeline import format_stage_filename, iter_stage_definitions


@dataclass(frozen=True)
class StageMetrics:
    prefix: int
    name: str
    file_name: str
    mae: float
    rmse: float
    psnr: float
    ssim: float | None
    reference_path: Path
    candidate_path: Path


@dataclass(frozen=True)
class StageDeltaMetrics:
    prefix: int
    name: str
    file_name: str
    reference_mae: float
    reference_rmse: float
    candidate_mae: float
    candidate_rmse: float


def _normalize_f32(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")

    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(image.astype(np.float32), 0.0, 1.0)

    raise ValueError(f"Unsupported image dtype: {image.dtype}")


def _load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image at: {path}")
    return image


def _compute_psnr(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _compute_ssim(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None

    return float(structural_similarity(reference, candidate, data_range=1.0)) # type: ignore


def _compute_mae_rmse(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    difference = candidate - reference
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(difference * difference))
    rmse = float(np.sqrt(mse))
    return mae, rmse, mse


def _format_metric(value: float, *, decimals: int = 6, tiny_threshold: float = 1.0e-6) -> str:
    if np.isinf(value):
        return "inf"
    if np.isnan(value):
        return "nan"

    absolute = abs(value)
    if 0.0 < absolute < tiny_threshold:
        return f"<{tiny_threshold:.{decimals}f}"

    return f"{value:.{decimals}f}"


def _crop_border(image: np.ndarray, crop_border_px: int) -> np.ndarray:
    if crop_border_px <= 0:
        return image

    height, width = image.shape
    if crop_border_px * 2 >= width or crop_border_px * 2 >= height:
        raise ValueError(
            f"Requested crop border {crop_border_px}px is too large for image shape {image.shape}"
        )

    return image[crop_border_px:height - crop_border_px, crop_border_px:width - crop_border_px]


def compare_stage_directories(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    crop_border_px: int = 0,
) -> list[StageMetrics]:
    if not reference_dir.exists() or not reference_dir.is_dir():
        raise ValueError(f"Reference directory does not exist: {reference_dir}")
    if not candidate_dir.exists() or not candidate_dir.is_dir():
        raise ValueError(f"Candidate directory does not exist: {candidate_dir}")

    results: list[StageMetrics] = []
    for stage in iter_stage_definitions():
        file_name = format_stage_filename(stage.prefix, stage.name)
        reference_path = reference_dir / file_name
        candidate_path = candidate_dir / file_name

        if not reference_path.exists():
            raise FileNotFoundError(f"Missing reference stage file: {reference_path}")
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing candidate stage file: {candidate_path}")

        reference = _normalize_f32(_load_grayscale(reference_path))
        candidate = _normalize_f32(_load_grayscale(candidate_path))

        if reference.shape != candidate.shape:
            raise ValueError(
                f"Shape mismatch for {file_name}: reference={reference.shape}, candidate={candidate.shape}"
            )

        reference = _crop_border(reference, crop_border_px)
        candidate = _crop_border(candidate, crop_border_px)

        mae, rmse, mse = _compute_mae_rmse(reference, candidate)
        psnr = _compute_psnr(mse)
        ssim = _compute_ssim(reference, candidate)

        results.append(
            StageMetrics(
                prefix=stage.prefix,
                name=stage.name,
                file_name=file_name,
                mae=mae,
                rmse=rmse,
                psnr=psnr,
                ssim=ssim,
                reference_path=reference_path,
                candidate_path=candidate_path,
            )
        )

    return results


def _load_stage_image_map(directory: Path, crop_border_px: int) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for stage in iter_stage_definitions():
        file_name = format_stage_filename(stage.prefix, stage.name)
        path = directory / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing stage file: {path}")

        image = _normalize_f32(_load_grayscale(path))
        images[stage.name] = _crop_border(image, crop_border_px)

    return images


def compare_stage_deltas(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    crop_border_px: int = 0,
) -> list[StageDeltaMetrics]:
    reference_images = _load_stage_image_map(reference_dir, crop_border_px)
    candidate_images = _load_stage_image_map(candidate_dir, crop_border_px)

    ordered_stages = list(iter_stage_definitions())
    results: list[StageDeltaMetrics] = []

    for index in range(1, len(ordered_stages)):
        previous_stage = ordered_stages[index - 1]
        stage = ordered_stages[index]

        reference_prev = reference_images[previous_stage.name]
        reference_current = reference_images[stage.name]
        candidate_prev = candidate_images[previous_stage.name]
        candidate_current = candidate_images[stage.name]

        if reference_prev.shape != reference_current.shape:
            raise ValueError(
                f"Reference stage shape mismatch between {previous_stage.name} and {stage.name}: "
                f"{reference_prev.shape} vs {reference_current.shape}"
            )
        if candidate_prev.shape != candidate_current.shape:
            raise ValueError(
                f"Candidate stage shape mismatch between {previous_stage.name} and {stage.name}: "
                f"{candidate_prev.shape} vs {candidate_current.shape}"
            )

        reference_mae, reference_rmse, _ = _compute_mae_rmse(reference_prev, reference_current)
        candidate_mae, candidate_rmse, _ = _compute_mae_rmse(candidate_prev, candidate_current)

        results.append(
            StageDeltaMetrics(
                prefix=stage.prefix,
                name=stage.name,
                file_name=format_stage_filename(stage.prefix, stage.name),
                reference_mae=reference_mae,
                reference_rmse=reference_rmse,
                candidate_mae=candidate_mae,
                candidate_rmse=candidate_rmse,
            )
        )

    return results


def format_metrics_table(rows: Iterable[StageMetrics]) -> str:
    table_rows = list(rows)
    header = f"{'stage':<25} {'MAE':>12} {'RMSE':>12} {'PSNR(dB)':>12} {'SSIM':>12}"
    divider = "-" * len(header)
    lines = [header, divider]

    for row in table_rows:
        mae_text = _format_metric(row.mae)
        rmse_text = _format_metric(row.rmse)
        psnr_text = _format_metric(row.psnr)
        ssim_text = "n/a" if row.ssim is None else _format_metric(row.ssim)
        lines.append(
            f"{row.file_name:<25} {mae_text:>12} {rmse_text:>12} {psnr_text:>12} {ssim_text:>12}"
        )

    if any(row.ssim is None for row in table_rows):
        lines.append("")
        lines.append("SSIM: n/a means scikit-image was not available in the active Python environment.")

    return "\n".join(lines)


def format_stage_delta_table(
    rows: Iterable[StageDeltaMetrics],
    *,
    reference_label: str = "reference",
    candidate_label: str = "candidate",
) -> str:
    table_rows = list(rows)
    reference_short = reference_label[:8]
    candidate_short = candidate_label[:8]
    header = (
        f"{'stage':<25} "
        f"{(reference_short + ' ΔMAE'):>12} {(reference_short + ' ΔRMSE'):>12} "
        f"{(candidate_short + ' ΔMAE'):>12} {(candidate_short + ' ΔRMSE'):>12}"
    )
    divider = "-" * len(header)
    lines = [header, divider]

    for row in table_rows:
        reference_mae_text = _format_metric(row.reference_mae)
        reference_rmse_text = _format_metric(row.reference_rmse)
        candidate_mae_text = _format_metric(row.candidate_mae)
        candidate_rmse_text = _format_metric(row.candidate_rmse)
        lines.append(
            f"{row.file_name:<25} "
            f"{reference_mae_text:>12} {reference_rmse_text:>12} "
            f"{candidate_mae_text:>12} {candidate_rmse_text:>12}"
        )

    lines.append("")
    lines.append("Δ metrics compare each stage image against the previous stage within the same pipeline.")
    lines.append("Low C++ Δ with higher reference Δ indicates a stage that is likely still close to passthrough.")
    return "\n".join(lines)
