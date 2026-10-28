"""Scatter painter: depth-ordered per-face texture rendering."""

import numpy as np

from litematic.render.isometric import _Scene
from litematic.render.templates import _shape_templates

_SHADES = (
    0.55 + 0.45 * (3 / np.sqrt(14)),
    0.55 + 0.45 * (2 / np.sqrt(14)),
    0.55 + 0.45 * (1 / np.sqrt(14)),
)
_MAGENTA = (255, 0, 255, 255)
_CUTOUT_ALPHA = 16
_COVER_ALPHA = 250


def _shade_face(array: np.ndarray, shade: float) -> np.ndarray:
    out = array.copy()
    rgb = np.rint(out[..., :3].astype(np.uint16) * shade)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def _paint(
    scene: _Scene, t: int, background: tuple[int, int, int, int]
) -> np.ndarray:
    """Render the scene onto an RGBA uint8 canvas at half-width scale t."""
    height = t * scene.span_h
    width = 2 * t * scene.span_w
    canvas = np.empty((height, width, 4), np.uint8)
    canvas[:] = background
    if len(scene.x) == 0:
        return canvas

    shaded = [
        tuple(_shade_face(face, shade) for shade, face in zip(_SHADES, slot))
        for slot in scene.slots
    ]
    magenta = tuple(
        _shade_face(np.full((16, 16, 4), _MAGENTA, np.uint8), shade)
        for shade in _SHADES
    )
    templates = _shape_templates(t)

    z_max = int(scene.z.max())
    y_max = int(scene.y.max())
    col_base = 2 * t * (scene.x - scene.z) + 2 * t * z_max
    row_base = t * (scene.x + scene.z) - 2 * t * scene.y + 2 * t * y_max
    depth = scene.x + scene.y + scene.z

    for value in np.unique(depth):
        layer = np.nonzero(depth == value)[0]
        for shape_value in np.unique(scene.shape[layer]):
            chosen = layer[scene.shape[layer] == shape_value]
            template = templates[int(shape_value)]
            rows0 = row_base[chosen]
            cols0 = col_base[chosen]
            block_src = [
                shaded[s] if s >= 0 else magenta for s in scene.face_slot[chosen]
            ]
            for face, (offsets, lut) in enumerate(
                (
                    (template.top, template.texel_top),
                    (template.east, template.texel_east),
                    (template.south, template.texel_south),
                )
            ):
                src = np.stack([slot[face] for slot in block_src])
                values = src[
                    np.arange(len(chosen))[:, None],
                    lut[None, :, 1],
                    lut[None, :, 0],
                ]
                rows = (rows0[:, None] + offsets[None, :, 0]).ravel()
                cols = (cols0[:, None] + offsets[None, :, 1]).ravel()
                flat = values.reshape(-1, 4)
                alpha = flat[:, 3]
                cover = alpha >= _COVER_ALPHA
                canvas[rows[cover], cols[cover]] = flat[cover]
                blend = (alpha >= _CUTOUT_ALPHA) & ~cover
                if blend.any():
                    _blend_pixels(canvas, rows[blend], cols[blend], flat[blend])
    return canvas


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
