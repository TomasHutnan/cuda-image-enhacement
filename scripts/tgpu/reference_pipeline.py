from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Iterable

import cv2
import numpy as np

from tgpu.paths import REPO_ROOT


@dataclass(frozen=True)
class CapturedStage:
    prefix: int
    name: str
    image: np.ndarray


@dataclass(frozen=True)
class StageDefinition:
    prefix: int
    name: str


_REFERENCE_FILE = REPO_ROOT / "reference" / "tescan_digital_resolution.py"

_STAGE_DEFINITIONS: tuple[StageDefinition, ...] = (
    StageDefinition(0, "input_normalized"),
    StageDefinition(10, "non_local_means"),
    StageDefinition(20, "unsharp_mask"),
    StageDefinition(30, "richardson_lucy"),
    StageDefinition(40, "histogram_stretch"),
    StageDefinition(90, "output"),
)


def _load_reference_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tescan_reference", _REFERENCE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load reference module from {_REFERENCE_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_to_f32(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    if np.issubdtype(image.dtype, np.floating):
        return image.astype(np.float32)
    raise ValueError(f"Unsupported image dtype: {image.dtype}")


def _to_saveable(image: np.ndarray, bit_depth: str) -> np.ndarray:
    if bit_depth not in {"u8", "u16"}:
        raise ValueError(f"Unsupported output bit depth: {bit_depth}")

    target_max = 255.0 if bit_depth == "u8" else 65535.0
    target_dtype = np.uint8 if bit_depth == "u8" else np.uint16

    if np.issubdtype(image.dtype, np.floating):
        clipped = np.clip(image, 0.0, 1.0)
        return np.rint(clipped * target_max).astype(target_dtype)

    if image.dtype == target_dtype:
        return image

    if image.dtype == np.uint8 and target_dtype == np.uint16:
        scale = 65535.0 / 255.0
        return np.rint(image.astype(np.float32) * scale).astype(np.uint16)

    if image.dtype == np.uint16 and target_dtype == np.uint8:
        scale = 255.0 / 65535.0
        return np.rint(image.astype(np.float32) * scale).astype(np.uint8)

    raise ValueError(f"Unsupported conversion from {image.dtype} to {target_dtype}")


def _to_reference_histogram_input(image: np.ndarray, rl_output_dtype: str) -> np.ndarray:
    if image.dtype == np.uint8 or image.dtype == np.uint16:
        return image

    if rl_output_dtype == "uint8":
        return _to_saveable(image, bit_depth="u8")

    return _to_saveable(image, bit_depth="u16")


def format_stage_filename(prefix: int, name: str) -> str:
    return f"{prefix:02d}_{name}.png"


def _is_stage_enabled(stage_name: str, only_stage: str | None) -> bool:
    if only_stage is None:
        return True
    return stage_name == only_stage


def run_reference_pipeline_with_stage_capture(
    input_image: np.ndarray,
    *,
    rl_output_dtype: str = "uint16",
    histogram_sat_percent: float = 0.5,
    only_stage: str | None = None,
) -> list[CapturedStage]:
    if input_image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")

    reference = _load_reference_module()

    stage_input = _normalize_to_f32(input_image)
    if only_stage is not None and only_stage not in set(iter_stage_names()):
        raise ValueError(f"Unsupported stage name for only_stage: {only_stage}")

    stage_nlm = stage_input
    if _is_stage_enabled("non_local_means", only_stage):
        stage_nlm = np.asarray(reference.non_local_means(stage_nlm))

    stage_unsharp = stage_nlm
    if _is_stage_enabled("unsharp_mask", only_stage):
        stage_unsharp = np.asarray(reference.unsharp_masking(stage_unsharp))

    stage_rl = stage_unsharp
    if _is_stage_enabled("richardson_lucy", only_stage):
        stage_rl = np.asarray(reference.richardson_lucy_deconvolution(stage_rl, output_dtype=rl_output_dtype))

    stage_hist = stage_rl
    if _is_stage_enabled("histogram_stretch", only_stage):
        stage_hist_input = _to_reference_histogram_input(np.asarray(stage_hist), rl_output_dtype)
        stage_hist = np.asarray(reference.histogram_stretch(stage_hist_input, sat_percent=histogram_sat_percent))

    return [
        CapturedStage(0, "input_normalized", stage_input),
        CapturedStage(10, "non_local_means", np.asarray(stage_nlm)),
        CapturedStage(20, "unsharp_mask", np.asarray(stage_unsharp)),
        CapturedStage(30, "richardson_lucy", np.asarray(stage_rl)),
        CapturedStage(40, "histogram_stretch", np.asarray(stage_hist)),
        CapturedStage(90, "output", np.asarray(stage_hist)),
    ]


def capture_reference_stages_from_file(
    input_path: Path,
    output_dir: Path,
    *,
    save_bit_depth: str = "u16",
    rl_output_dtype: str = "uint16",
    histogram_sat_percent: float = 0.5,
    only_stage: str | None = None,
) -> list[Path]:
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {input_path}")
    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel grayscale image: {input_path}")

    stages = run_reference_pipeline_with_stage_capture(
        image,
        rl_output_dtype=rl_output_dtype,
        histogram_sat_percent=histogram_sat_percent,
        only_stage=only_stage,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for stage in stages:
        stage_path = output_dir / format_stage_filename(stage.prefix, stage.name)
        saveable = _to_saveable(stage.image, bit_depth=save_bit_depth)
        if not cv2.imwrite(str(stage_path), saveable):
            raise RuntimeError(f"Failed to write stage image: {stage_path}")
        written_paths.append(stage_path)

    return written_paths


def iter_stage_names() -> Iterable[str]:
    return (stage.name for stage in _STAGE_DEFINITIONS)


def iter_stage_definitions() -> Iterable[StageDefinition]:
    return _STAGE_DEFINITIONS
