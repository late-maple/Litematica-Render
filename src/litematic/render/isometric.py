"""Isometric rendering: shape classification and pixel templates."""

from typing import NamedTuple, Sequence

import numpy as np

from litematic.parse.model import BlockState, Litematic, Region
from litematic.render.pack import Textures

SHAPE_CUBE = 0
SHAPE_SLAB_BOTTOM = 1
SHAPE_SLAB_TOP = 2

_TEX_SIZE = 16
_AIR_NAMES = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
_Y_SLAB = 64
_OPAQUE_ALPHA = 250.0


class _Template(NamedTuple):
    """Pixel offsets and texel lookup for one shape at one scale."""

    top: np.ndarray  # (P, 2) int16 (dy, dx)
    east: np.ndarray  # (P, 2) int16
    south: np.ndarray  # (P, 2) int16
    texel_top: np.ndarray  # (P, 2) uint8
    texel_east: np.ndarray  # (P, 2) uint8
    texel_south: np.ndarray  # (P, 2) uint8


def _shape_of(state: BlockState) -> int:
    """Map a block state to cube / bottom slab / top slab (double slab is a cube)."""
    path = state.name.split(":", 1)[-1]
    if not path.endswith("_slab"):
        return SHAPE_CUBE
    slab_type = state.properties.get("type", "bottom")
    if slab_type == "top":
        return SHAPE_SLAB_TOP
    if slab_type == "double":
        return SHAPE_CUBE
    return SHAPE_SLAB_BOTTOM


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
    return {
        SHAPE_CUBE: cube,
        SHAPE_SLAB_BOTTOM: slab_bottom,
        SHAPE_SLAB_TOP: slab_top,
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


class _Scene(NamedTuple):
    """Visible blocks after culling and yaw rotation, min-corner at origin."""

    x: np.ndarray  # int32
    y: np.ndarray  # int32
    z: np.ndarray  # int32
    shape: np.ndarray  # uint8
    tex_ok: np.ndarray  # bool
    span_w: int  # canvas width in pixels = 2t * span_w at scale t
    span_h: int  # canvas height in pixels = t * span_h at scale t


def _compose(schem: Litematic, yaw: int, textures: Textures) -> _Scene:
    """Collect visible blocks from all regions into rotated world coords."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    shapes: list[np.ndarray] = []
    oks: list[np.ndarray] = []
    for name in sorted(schem.regions):
        region = schem.regions[name]
        luts = _palette_luts(region.palette, textures)
        cells = _visible_cells(region, luts)
        if cells is None:
            continue
        cx, cy, cz, cshape, cok = cells
        pos = region.position
        xs.append(cx + pos.x)
        ys.append(cy + pos.y)
        zs.append(cz + pos.z)
        shapes.append(cshape)
        oks.append(cok)
    if not xs:
        return _Scene(
            np.zeros(0, np.int32),
            np.zeros(0, np.int32),
            np.zeros(0, np.int32),
            np.zeros(0, np.uint8),
            np.zeros(0, bool),
            2,
            4,
        )
    x = np.concatenate(xs).astype(np.int32)
    y = np.concatenate(ys).astype(np.int32)
    z = np.concatenate(zs).astype(np.int32)
    shape = np.concatenate(shapes)
    tex_ok = np.concatenate(oks)
    xr = x - int(x.min())
    zr = z - int(z.min())
    yr = y - int(y.min())
    rx, rz = _yaw_rot(xr, zr, yaw, int(xr.max()) + 1, int(zr.max()) + 1)
    span_w = int(rx.max() + rz.max() + 2)
    span_h = int(rx.max() + rz.max() + 2 * yr.max() + 4)
    return _Scene(rx, yr, rz, shape, tex_ok, span_w, span_h)


def _yaw_rot(
    x: np.ndarray, z: np.ndarray, yaw: int, w: int, d: int
) -> tuple[np.ndarray, np.ndarray]:
    if yaw == 1:
        return z, (w - 1) - x
    if yaw == 2:
        return (w - 1) - x, (d - 1) - z
    if yaw == 3:
        return (d - 1) - z, x
    return x, z


def _palette_luts(
    palette: Sequence[BlockState], textures: Textures
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(palette)
    is_air = np.zeros(count, bool)
    occludes = np.zeros(count, bool)
    shapes = np.zeros(count, np.uint8)
    tex_ok = np.zeros(count, bool)
    for i, state in enumerate(palette):
        if state.name in _AIR_NAMES:
            is_air[i] = True
            continue
        shape = _shape_of(state)
        shapes[i] = shape
        faces = textures.faces(state.name, state.properties)
        if faces is None:
            occludes[i] = shape == SHAPE_CUBE
            continue
        tex_ok[i] = True
        alpha = float(np.mean([face[..., 3].mean() for face in faces]))
        occludes[i] = alpha >= _OPAQUE_ALPHA and shape == SHAPE_CUBE
    return is_air, occludes, shapes, tex_ok


def _visible_cells(
    region: Region, luts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    is_air, occludes, shapes, tex_ok = luts
    idx = region.block_indices
    sx, sy, sz = idx.shape
    out_x: list[np.ndarray] = []
    out_y: list[np.ndarray] = []
    out_z: list[np.ndarray] = []
    out_shape: list[np.ndarray] = []
    out_ok: list[np.ndarray] = []
    for y0 in range(0, sy, _Y_SLAB):
        y1 = min(y0 + _Y_SLAB, sy)
        body = idx[:, y0:y1, :]
        occ_body = occludes[body]
        present = ~is_air[body]
        rows = idx[:, y0 : min(y1 + 1, sy), :]
        occ_rows = occludes[rows]
        upper = occ_rows[:, 1:, :]
        if y1 == sy:
            upper = np.concatenate(
                [upper, np.zeros((sx, 1, sz), bool)], axis=1
            )
        occ_x = np.zeros_like(occ_body)
        occ_x[:-1, :, :] = occ_body[1:, :, :]
        occ_z = np.zeros_like(occ_body)
        occ_z[:, :, :-1] = occ_body[:, :, 1:]
        visible = present & ~(upper & occ_x & occ_z)
        if not visible.any():
            continue
        lx, ly, lz = np.nonzero(visible)
        picked = body[visible]
        out_x.append(lx)
        out_y.append(ly + y0)
        out_z.append(lz)
        out_shape.append(shapes[picked])
        out_ok.append(tex_ok[picked])
    if not out_x:
        return None
    return (
        np.concatenate(out_x),
        np.concatenate(out_y),
        np.concatenate(out_z),
        np.concatenate(out_shape),
        np.concatenate(out_ok),
    )
