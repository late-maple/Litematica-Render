"""Scatter painter: depth-ordered per-face texture rendering."""

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.isometric import _Scene

_SHADES = (
    0.55 + 0.45 * (3 / np.sqrt(14)),
    0.55 + 0.45 * (2 / np.sqrt(14)),
    0.55 + 0.45 * (1 / np.sqrt(14)),
)
_MAGENTA = (255, 0, 255, 255)
_CUTOUT_ALPHA = 16
_COVER_ALPHA = 250
_FUSE_PIXELS = 1 << 25


def _shade_face(array: np.ndarray, shade: float) -> np.ndarray:
    out = array.copy()
    rgb = np.rint(out[..., :3].astype(np.uint16) * shade)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def _paint(
    scene: _Scene, t: int, background: tuple[int, int, int, int]
) -> np.ndarray:
    """Placeholder: the v2 quad painter lands with plan Task 7."""
    raise RenderError(
        "paint: quad painter is not wired yet (v2 migration, plan Task 7)"
    )


def _blend_pixels(
    canvas: np.ndarray, rows: np.ndarray, cols: np.ndarray, src: np.ndarray
) -> None:
    dst = canvas[rows, cols].astype(np.uint16)
    src_u = src.astype(np.uint16)
    alpha = src_u[:, 3:4]
    out_rgb = np.rint((src_u[:, :3] * alpha + dst[:, :3] * (255 - alpha)) / 255)
    out_alpha = np.rint(src_u[:, 3:4] + dst[:, 3:4] * (255 - alpha) / 255)
    blended = np.concatenate(
        [np.clip(out_rgb, 0, 255), np.clip(out_alpha, 0, 255)], axis=1
    ).astype(np.uint8)
    canvas[rows, cols] = blended


def _choose_scale(
    span_w: int, span_h: int, max_size: int, supersample: int
) -> tuple[int, int]:
    """Pick tile scale and supersample factor within the canvas pixel fuse."""
    for ss in range(supersample, 0, -1):
        t = max(1, min(max_size * ss // (2 * span_w), max_size * ss // span_h))
        width = 2 * t * span_w
        height = t * span_h
        fits = width <= max_size * ss and height <= max_size * ss
        if (fits or ss == 1) and width * height <= _FUSE_PIXELS:
            return t, ss
    raise RenderError(
        f"scale: spans {span_w}x{span_h} exceed canvas budget at "
        f"max_size={max_size}"
    )


def _downsample(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Box-filter downsample RGBA uint8 with alpha-weighted color."""
    in_h, in_w = img.shape[:2]
    if out_w == in_w and out_h == in_h:
        return img.copy()
    rows = np.minimum(np.arange(in_h) * out_h // in_h, out_h - 1)
    cols = np.minimum(np.arange(in_w) * out_w // in_w, out_w - 1)
    flat = (rows[:, None] * out_w + cols[None, :]).ravel()
    size = out_h * out_w
    alpha = img[..., 3].ravel().astype(np.float64) / 255.0
    weight = np.bincount(flat, weights=alpha, minlength=size)
    count = np.bincount(flat, minlength=size).astype(np.float64)
    out_a = np.rint(weight / count * 255.0)
    covered = (weight > 0) & (out_a > 0)
    out = np.zeros((size, 4), np.uint8)
    out[:, 3] = np.clip(out_a, 0, 255).astype(np.uint8)
    for channel in range(3):
        values = img[..., channel].ravel().astype(np.float64)
        summed = np.bincount(flat, weights=values * alpha, minlength=size)
        mean = np.zeros(size)
        mean[covered] = summed[covered] / weight[covered]
        out[:, channel] = np.clip(np.rint(mean), 0, 255).astype(np.uint8)
    return out.reshape(out_h, out_w, 4)
