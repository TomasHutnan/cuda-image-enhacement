from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tgpu.reference_pipeline import format_stage_filename, iter_stage_definitions

_DEFAULT_W = 1600
_DEFAULT_H = 900
_WINDOW_NAME = "tgpu stage viewer"
_KEY_QUIT = (27, ord("q"), ord("Q"))
_KEY_PREV = (81, 2424832, ord("a"), ord("A"))
_KEY_NEXT = (83, 2555904, ord("d"), ord("D"))


@dataclass(frozen=True)
class StagePair:
    prefix: int
    name: str
    file_name: str
    first_path: Path
    second_path: Path


def _load_grayscale_f32(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale stage image: {path}")
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif np.issubdtype(image.dtype, np.floating):
        return np.clip(image.astype(np.float32), 0.0, 1.0)
    raise ValueError(f"Unsupported grayscale stage image dtype: {image.dtype} ({path})")


def _display_bgr_from_f32(image: np.ndarray) -> np.ndarray:
    image_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)


def _load_display_bgr(path: Path) -> np.ndarray:
    return _display_bgr_from_f32(_load_grayscale_f32(path))


def _make_diff_display_bgr(first_path: Path, second_path: Path) -> np.ndarray:
    first = _load_grayscale_f32(first_path)
    second = _load_grayscale_f32(second_path)

    if first.shape != second.shape:
        raise ValueError(f"Mismatched stage image shapes: {first_path} vs {second_path}")

    diff = np.abs(first - second)
    max_diff = float(diff.max())
    if max_diff > 0.0:
        diff = diff / max_diff
    return _display_bgr_from_f32(diff)


def _fit_inside(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if nw == w and nh == h:
        return image
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)


def _fit_to_frame(image: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fit image into a (w x h) black frame, centered, preserving aspect ratio."""
    fitted = _fit_inside(image, w, h)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    y = (h - fitted.shape[0]) // 2
    x = (w - fitted.shape[1]) // 2
    frame[y:y + fitted.shape[0], x:x + fitted.shape[1]] = fitted
    return frame


def _is_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1.0
    except cv2.error:
        return True


def _window_size(window_name: str) -> tuple[int, int]:
    try:
        _, _, w, h = cv2.getWindowImageRect(window_name)
        if w > 0 and h > 0:
            return w, h
    except cv2.error:
        pass
    return _DEFAULT_W, _DEFAULT_H


def _text(canvas: np.ndarray, text: str, pos: tuple[int, int], scale: float, color: tuple, thickness: int = 1) -> None:
    cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _make_pair_canvas(pair: StagePair, first_name: str, second_name: str, index: int, total: int) -> np.ndarray:
    panel_w = (_DEFAULT_W - 40) // 3
    panel_h = _DEFAULT_H - 84

    def make_panel(path: Path, label: str) -> np.ndarray:
        img = _fit_inside(_load_display_bgr(path), panel_w, panel_h)
        ih, iw = img.shape[:2]
        panel = np.zeros((panel_h + 44, panel_w, 3), dtype=np.uint8)
        panel[44 + (panel_h - ih) // 2:44 + (panel_h + ih) // 2, (panel_w - iw) // 2:(panel_w + iw) // 2] = img
        _text(panel, label, (10, 28), 0.7, (230, 230, 230), 2)
        return panel

    def make_diff_panel() -> np.ndarray:
        img = _fit_inside(_make_diff_display_bgr(pair.first_path, pair.second_path), panel_w, panel_h)
        ih, iw = img.shape[:2]
        panel = np.zeros((panel_h + 44, panel_w, 3), dtype=np.uint8)
        panel[44 + (panel_h - ih) // 2:44 + (panel_h + ih) // 2, (panel_w - iw) // 2:(panel_w + iw) // 2] = img
        _text(panel, "diff |py-cpp| (auto)", (10, 28), 0.62, (230, 230, 230), 2)
        return panel

    left = make_panel(pair.first_path, first_name)
    mid = make_panel(pair.second_path, second_name)
    right = make_diff_panel()
    gap = np.zeros((left.shape[0], 20, 3), dtype=np.uint8)
    canvas = np.concatenate([left, gap, mid, gap, right], axis=1)
    _text(canvas, f"{pair.file_name}  ({index + 1}/{total})", (10, canvas.shape[0] - 28), 0.65, (0, 210, 255), 2)
    _text(canvas, "Left/Right: prev/next stage  |  Q / Esc: quit", (10, canvas.shape[0] - 8), 0.5, (180, 180, 180))
    return canvas


def _make_grid_canvas(pairs: list[StagePair], first_name: str, second_name: str, slot_w: int = 300, slot_h: int = 220) -> np.ndarray:
    STAGE_H, BAR_H, FOOTER_H, GAP = 20, 32, 28, 10
    n = len(pairs)
    col_h = STAGE_H + (BAR_H + slot_h) * 3
    canvas = np.zeros((col_h + FOOTER_H, n * slot_w + (n - 1) * GAP, 3), dtype=np.uint8)

    for col, pair in enumerate(pairs):
        x = col * (slot_w + GAP)
        _text(canvas, pair.file_name, (x + 4, STAGE_H - 4), 0.42, (0, 210, 255))

        r1_img_y = STAGE_H + BAR_H
        _text(canvas, first_name, (x + 4, STAGE_H + BAR_H - 10), 0.42, (220, 220, 220))
        canvas[r1_img_y:r1_img_y + slot_h, x:x + slot_w] = _fit_to_frame(_load_display_bgr(pair.first_path), slot_w, slot_h)

        r2_img_y = r1_img_y + slot_h + BAR_H
        _text(canvas, second_name, (x + 4, r1_img_y + slot_h + BAR_H - 10), 0.42, (220, 220, 220))
        canvas[r2_img_y:r2_img_y + slot_h, x:x + slot_w] = _fit_to_frame(_load_display_bgr(pair.second_path), slot_w, slot_h)

        r3_img_y = r2_img_y + slot_h + BAR_H
        _text(canvas, "diff |py-cpp| (auto)", (x + 4, r2_img_y + slot_h + BAR_H - 10), 0.42, (220, 220, 220))
        canvas[r3_img_y:r3_img_y + slot_h, x:x + slot_w] = _fit_to_frame(
            _make_diff_display_bgr(pair.first_path, pair.second_path),
            slot_w,
            slot_h,
        )

    _text(canvas, "Q or Esc: quit", (4, canvas.shape[0] - 8), 0.45, (180, 180, 180))
    return canvas


def _collect_stage_pairs(first_dir: Path, second_dir: Path, only_stage: str | None) -> list[StagePair]:
    for d in (first_dir, second_dir):
        if not d.is_dir():
            raise ValueError(f"Stage directory does not exist: {d}")

    all_stages = list(iter_stage_definitions())
    if only_stage is not None and only_stage not in {s.name for s in all_stages}:
        raise ValueError(f"Unsupported stage name: {only_stage}")

    pairs = []
    for stage in all_stages:
        if only_stage is not None and stage.name != only_stage:
            continue
        file_name = format_stage_filename(stage.prefix, stage.name)
        first_path, second_path = first_dir / file_name, second_dir / file_name
        for p in (first_path, second_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing stage file: {p}")
        pairs.append(StagePair(stage.prefix, stage.name, file_name, first_path, second_path))

    if not pairs:
        raise ValueError("No matching stage files found")
    return pairs


def _run_pair_loop(window_name: str, pairs: list[StagePair], first_label: str, second_label: str) -> None:
    index, last_index, last_size = 0, -1, (-1, -1)
    base_canvas: np.ndarray | None = None

    while not _is_closed(window_name):
        if index != last_index:
            base_canvas = _make_pair_canvas(pairs[index], first_label, second_label, index, len(pairs))
            last_index, last_size = index, (-1, -1)

        size = _window_size(window_name)
        if size != last_size:
            cv2.imshow(window_name, _fit_to_frame(base_canvas, *size))  # type: ignore[arg-type]
            last_size = size

        key = cv2.waitKeyEx(30)
        if key in _KEY_QUIT:
            break
        if key in _KEY_PREV:
            index = (index - 1) % len(pairs)
        if key in _KEY_NEXT:
            index = (index + 1) % len(pairs)


def _run_grid_loop(window_name: str, base_canvas: np.ndarray) -> None:
    last_size = (-1, -1)

    while not _is_closed(window_name):
        size = _window_size(window_name)
        if size != last_size:
            cv2.imshow(window_name, _fit_to_frame(base_canvas, *size))
            last_size = size

        if cv2.waitKeyEx(30) in _KEY_QUIT:
            break


def show_stage_viewer(
    first_dir: Path,
    second_dir: Path,
    *,
    first_label: str = "first",
    second_label: str = "second",
    mode: str = "pair",
    only_stage: str | None = None,
) -> None:
    pairs = _collect_stage_pairs(first_dir, second_dir, only_stage)

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WINDOW_NAME, _DEFAULT_W, _DEFAULT_H)

    try:
        if mode == "grid":
            _run_grid_loop(_WINDOW_NAME, _make_grid_canvas(pairs, first_label, second_label))
        else:
            _run_pair_loop(_WINDOW_NAME, pairs, first_label, second_label)
    finally:
        if not _is_closed(_WINDOW_NAME):
            cv2.destroyWindow(_WINDOW_NAME)
