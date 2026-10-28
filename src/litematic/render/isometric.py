"""Scene composition: culling, yaw rotation, span computation."""

from typing import NamedTuple, Sequence

import numpy as np

from litematic.parse.model import BlockState, Litematic, Region
from litematic.render.pack import Textures
from litematic.render.templates import SHAPE_CUBE, _shape_of

_AIR_NAMES = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
_Y_SLAB = 64
_OPAQUE_ALPHA = 250.0


class _Scene(NamedTuple):
    """Visible blocks after culling and yaw rotation, min-corner at origin."""

    x: np.ndarray  # int32
    y: np.ndarray  # int32
    z: np.ndarray  # int32
    shape: np.ndarray  # uint8
    tex_ok: np.ndarray  # bool
    span_w: int  # canvas width in pixels = 2t * span_w at scale t
    span_h: int  # canvas height in pixels = t * span_h at scale t
    face_slot: np.ndarray  # int32, index into slots; -1 renders magenta
    slots: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]


def _compose(schem: Litematic, yaw: int, textures: Textures) -> _Scene:
    """Collect visible blocks from all regions into rotated scene coords."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    shapes: list[np.ndarray] = []
    oks: list[np.ndarray] = []
    slot_ids: list[np.ndarray] = []
    slots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for name in sorted(schem.regions):
        region = schem.regions[name]
        luts = _palette_luts(region.palette, textures, slots)
        cells = _visible_cells(region, luts)
        if cells is None:
            continue
        cx, cy, cz, cshape, cok, cslot = cells
        pos = region.position
        xs.append(cx + pos.x)
        ys.append(cy + pos.y)
        zs.append(cz + pos.z)
        shapes.append(cshape)
        oks.append(cok)
        slot_ids.append(cslot)
    if not xs:
        return _Scene(
            np.zeros(0, np.int32),
            np.zeros(0, np.int32),
            np.zeros(0, np.int32),
            np.zeros(0, np.uint8),
            np.zeros(0, bool),
            2,
            4,
            np.zeros(0, np.int32),
            (),
        )
    x = np.concatenate(xs).astype(np.int32)
    y = np.concatenate(ys).astype(np.int32)
    z = np.concatenate(zs).astype(np.int32)
    shape = np.concatenate(shapes)
    tex_ok = np.concatenate(oks)
    face_slot = np.concatenate(slot_ids).astype(np.int32)
    xr = x - int(x.min())
    zr = z - int(z.min())
    yr = y - int(y.min())
    rx, rz = _yaw_rot(xr, zr, yaw, int(xr.max()) + 1, int(zr.max()) + 1)
    span_w = int(rx.max() + rz.max() + 2)
    span_h = int(rx.max() + rz.max() + 2 * yr.max() + 4)
    return _Scene(
        rx, yr, rz, shape, tex_ok, span_w, span_h, face_slot, tuple(slots)
    )


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


class _Luts(NamedTuple):
    """Per-palette-entry lookup arrays for culling and shading."""

    is_air: np.ndarray  # bool
    occludes: np.ndarray  # bool
    shapes: np.ndarray  # uint8
    tex_ok: np.ndarray  # bool
    slot_lut: np.ndarray  # int32, -1 for pack misses


class _Cells(NamedTuple):
    """Visible cells of one region in region-local coordinates."""

    x: np.ndarray  # int
    y: np.ndarray  # int
    z: np.ndarray  # int
    shape: np.ndarray  # uint8
    tex_ok: np.ndarray  # bool
    slot: np.ndarray  # int32


def _palette_luts(
    palette: Sequence[BlockState],
    textures: Textures,
    slots: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> _Luts:
    count = len(palette)
    is_air = np.zeros(count, bool)
    occludes = np.zeros(count, bool)
    shapes = np.zeros(count, np.uint8)
    tex_ok = np.zeros(count, bool)
    slot_lut = np.full(count, -1, np.int32)
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
        slot_lut[i] = len(slots)
        slots.append(faces)
        alpha = float(np.mean([face[..., 3].mean() for face in faces]))
        occludes[i] = alpha >= _OPAQUE_ALPHA and shape == SHAPE_CUBE
    return _Luts(is_air, occludes, shapes, tex_ok, slot_lut)


def _visible_cells(region: Region, luts: _Luts) -> _Cells | None:
    is_air, occludes, shapes, tex_ok, slot_lut = luts
    idx = region.block_indices
    sx, sy, sz = idx.shape
    out_x: list[np.ndarray] = []
    out_y: list[np.ndarray] = []
    out_z: list[np.ndarray] = []
    out_shape: list[np.ndarray] = []
    out_slot: list[np.ndarray] = []
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
        out_slot.append(slot_lut[picked])
    if not out_x:
        return None
    return _Cells(
        np.concatenate(out_x),
        np.concatenate(out_y),
        np.concatenate(out_z),
        np.concatenate(out_shape),
        np.concatenate(out_ok),
        np.concatenate(out_slot),
    )
