"""Isometric shape templates: classification and pixel rasterization."""

from typing import NamedTuple

import numpy as np

from litematic.parse.model import BlockState

SHAPE_CUBE = 0
SHAPE_SLAB_BOTTOM = 1
SHAPE_SLAB_TOP = 2
SHAPE_FLAT = 3

_TEX_SIZE = 16

_FLAT_NAMES = {
    "rail",
    "repeater",
    "comparator",
    "redstone_wire",
    "snow",
    "string",
    "tripwire",
    "tripwire_hook",
    "frogspawn",
}
_FLAT_SUFFIXES = ("_rail", "_pressure_plate", "_carpet")


class _Template(NamedTuple):
    """Pixel offsets and texel lookup for one shape at one scale."""

    top: np.ndarray  # (P, 2) int16 (dy, dx)
    east: np.ndarray  # (P, 2) int16
    south: np.ndarray  # (P, 2) int16
    texel_top: np.ndarray  # (P, 2) uint8
    texel_east: np.ndarray  # (P, 2) uint8
    texel_south: np.ndarray  # (P, 2) uint8


def _shape_of(state: BlockState) -> int:
    """Map a block state to cube / slab / flat ground shapes."""
    path = state.name.split(":", 1)[-1]
    if path.endswith("_slab"):
        slab_type = state.properties.get("type", "bottom")
        if slab_type == "top":
            return SHAPE_SLAB_TOP
        if slab_type == "double":
            return SHAPE_CUBE
        return SHAPE_SLAB_BOTTOM
    if path in _FLAT_NAMES or path.endswith(_FLAT_SUFFIXES):
        return SHAPE_FLAT
    return SHAPE_CUBE


def _shape_templates(t: int) -> dict[int, _Template]:
    """Build the three shape templates for half-width scale t."""
    cube = _template(
        t, top_v=t, east_top=3 * t, south_top=t, side_h=2 * t, side_v=(0, 16)
    )
    slab_bottom = _template(
        t, top_v=2 * t, east_top=4 * t, south_top=2 * t, side_h=t, side_v=(8, 16)
    )
    slab_top = _template(
        t, top_v=t, east_top=3 * t, south_top=t, side_h=t, side_v=(0, 8)
    )
    empty = np.zeros((0, 2), np.int16)
    empty_texel = np.zeros((0, 2), np.uint8)
    flat = _Template(
        top=cube.top,
        east=empty,
        south=empty,
        texel_top=cube.texel_top,
        texel_east=empty_texel,
        texel_south=empty_texel,
    )
    return {
        SHAPE_CUBE: cube,
        SHAPE_SLAB_BOTTOM: slab_bottom,
        SHAPE_SLAB_TOP: slab_top,
        SHAPE_FLAT: flat,
    }


def _template(
    t: int,
    top_v: float,
    east_top: float,
    south_top: float,
    side_h: float,
    side_v: tuple[int, int],
) -> _Template:
    """Build one shape's template."""
    size = 4 * t
    cols = np.arange(size, dtype=np.float64) + 0.5
    rows = np.arange(size, dtype=np.float64) + 0.5
    u = np.tile(cols, size)  # column coordinate, row-major flat grid
    v = np.repeat(rows, size)

    half = 2 * t
    top_w = half - 2 * np.abs(v - top_v)
    in_top = (np.abs(u - half) <= top_w) & (top_w > 0)
    east_a = east_top - u / 2
    in_east = (u >= half) & (u <= 2 * half) & (v >= east_a) & (v <= east_a + side_h)
    south_a = south_top + u / 2
    in_south = (u >= 0) & (u <= half) & (v >= south_a) & (v <= south_a + side_h)

    def texel_clip(values: np.ndarray) -> np.ndarray:
        return np.clip(np.floor(values), 0, _TEX_SIZE - 1).astype(np.uint8)

    top_denom = 2 * np.maximum(top_w, 1e-9)
    top_tx = texel_clip((u - (half - top_w)) / top_denom * _TEX_SIZE)
    top_ty = texel_clip((v - (top_v - t)) / (2 * t) * _TEX_SIZE)
    side_span = side_v[1] - side_v[0]
    east_tx = texel_clip((u - half) / (2 * t) * _TEX_SIZE)
    east_ty = texel_clip(side_v[0] + (v - east_a) / side_h * side_span)
    south_tx = texel_clip(u / (2 * t) * _TEX_SIZE)
    south_ty = texel_clip(side_v[0] + (v - south_a) / side_h * side_span)

    def face(mask, tx, ty):
        idx = np.nonzero(mask)[0]
        pixels = np.stack([idx // size, idx % size], axis=1).astype(np.int16)
        texels = np.stack([tx[mask], ty[mask]], axis=1).astype(np.uint8)
        return pixels, texels

    top_px, top_tex = face(in_top, top_tx, top_ty)
    east_px, east_tex = face(in_east, east_tx, east_ty)
    south_px, south_tex = face(in_south, south_tx, south_ty)
    return _Template(
        top=top_px,
        east=east_px,
        south=south_px,
        texel_top=top_tex,
        texel_east=east_tex,
        texel_south=south_tex,
    )
