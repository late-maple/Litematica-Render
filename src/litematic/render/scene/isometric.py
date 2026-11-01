"""Scene composition: quad expansion, camera filtering, and vertex AO."""

import math
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from litematic.parse.model import Litematic
from litematic.render.entity import (
    armor_stand_quads,
    entity_props,
    item_entity_quads,
    item_frame_quads,
)
from litematic.render.errors import RenderError
from litematic.render.pack.resolve import Resolver
from litematic.render.quads import (
    AO_LEVELS,
    CULL_NONE,
    TIER_DOWN,
    TIER_EAST,
    TIER_NORMALS,
    TIER_NORTH,
    TIER_SOUTH,
    TIER_UP,
    TIER_WEST,
)
from litematic.render.scene.view import (
    ISO_SLANT_DEG,
    ViewAxes,
    project_bounds,
    view_axes,
)

_AIR_NAMES = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
_SPAN_SNAP = 1e-9
_WATER_FAMILY = {"minecraft:water", "minecraft:bubble_column"}
_Y_SLAB = 64

# Face normal vectors per tier in block space; AO samples sit one step out
# along the normal and one step along each tangent sign.
_TIER_NORMALS = TIER_NORMALS.astype(np.int64)

# Per tier: normal axis then the two tangent axes for AO corner sampling.
_TIER_AXES = np.zeros((6, 3), np.int64)
_TIER_AXES[TIER_DOWN] = (1, 0, 2)
_TIER_AXES[TIER_UP] = (1, 0, 2)
_TIER_AXES[TIER_NORTH] = (2, 0, 1)
_TIER_AXES[TIER_SOUTH] = (2, 0, 1)
_TIER_AXES[TIER_WEST] = (0, 1, 2)
_TIER_AXES[TIER_EAST] = (0, 1, 2)


class _Scene(NamedTuple):
    """Camera-facing quads with vertex AO, world coords, and camera triad."""

    x: np.ndarray  # int32 per-quad block x
    y: np.ndarray  # int32 per-quad block y
    z: np.ndarray  # int32 per-quad block z
    quad_vtx: np.ndarray  # (Q, 4, 3) f32 block-local texels
    quad_uv: np.ndarray  # (Q, 4, 2) f32 texel coords
    quad_ao: np.ndarray  # (Q, 4) f32 per-vertex AO multipliers
    quad_shade: np.ndarray  # (Q,) u1 world-facing tier
    quad_dbl: np.ndarray  # (Q,) bool double-sided flag
    texkey_ids: np.ndarray  # (Q,) i32 index into texkeys
    texkeys: dict[str, np.ndarray]  # unique texkey -> (h, w, 4) array
    span_w: int  # canvas width in pixels = t * span_w at scale t
    span_h: int  # canvas height in pixels = t * span_h at scale t
    axes: ViewAxes  # camera triad


class _Luts(NamedTuple):
    """Per-palette-entry lookups for culling and quad expansion."""

    is_air: np.ndarray  # bool
    occludes: np.ndarray  # bool
    pool: np.ndarray | None  # (P,) QUAD_DTYPE, None when every entry is air
    offsets: np.ndarray  # (E + 1,) i8 start of each entry's rows in pool
    texkey_ids: np.ndarray  # (P,) i32 per-pool-row texkey id
    cull_class: np.ndarray  # (P,) i32 pair class, negative entries never pair


class _Arena(NamedTuple):
    """Preallocated scene columns filled slab by slab."""

    x: np.ndarray  # (K,) i32 world block x per quad
    y: np.ndarray  # (K,) i32 world block y per quad
    z: np.ndarray  # (K,) i32 world block z per quad
    vtx: np.ndarray  # (K, 4, 3) f32 block-local texels
    uv: np.ndarray  # (K, 4, 2) f32
    ao: np.ndarray  # (K, 4) f32
    shade: np.ndarray  # (K,) u1 world-facing tier
    dbl: np.ndarray  # (K,) bool
    texkey: np.ndarray  # (K,) i32


def _new_arena(capacity: int) -> _Arena:
    """Allocate the exact scene columns for the counted quad total."""
    return _Arena(
        np.zeros(capacity, np.int32),
        np.zeros(capacity, np.int32),
        np.zeros(capacity, np.int32),
        np.zeros((capacity, 4, 3), np.float32),
        np.zeros((capacity, 4, 2), np.float32),
        np.zeros((capacity, 4), np.float32),
        np.zeros(capacity, np.uint8),
        np.zeros(capacity, bool),
        np.zeros(capacity, np.int32),
    )


def _compose(
    schem: Litematic,
    rotation: float,
    slant: float,
    textures: Resolver,
    entities: bool = True,
) -> _Scene:
    """Collect visible blocks and entities from all regions into a scene."""
    try:
        return _compose_inner(schem, rotation, slant, textures, entities)
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError(f"compose: scene composition failed: {exc}") from exc


def _compose_inner(
    schem: Litematic,
    rotation: float,
    slant: float,
    textures: Resolver,
    entities: bool,
) -> _Scene:
    """Expand every region's visible cells into camera-filtered quads.

    Pass A stashes per-slab visible cells and counts the exact
    camera-facing quad total; pass B expands each slab straight into the
    preallocated arena through a rolling occlusion window, so no
    intermediate list of slab pieces is ever held.
    """
    axes = view_axes(rotation, slant)
    tier_visible = np.zeros(6, bool)
    tier_visible[list(axes.tiers)] = True
    key_ids: dict[str, int] = {}
    key_arrays: dict[str, np.ndarray] = {}
    plan = []
    capacity = 0
    for name in sorted(schem.regions):
        region = schem.regions[name]
        luts = _palette_luts(region.palette, textures, key_ids, key_arrays)
        if luts.pool is None:
            continue
        cam_rows = _camera_row_counts(luts, tier_visible)
        stash = list(_visible_slabs(region, luts))
        capacity += sum(
            int(cam_rows[picked.astype(np.int64)].sum())
            for (*_, picked) in stash
        )
        plan.append((region, luts, stash))
    entity_rows = (
        _entity_rows(schem, textures, key_ids, key_arrays) if entities else None
    )
    entity_count = 0 if entity_rows is None else len(entity_rows[0])
    if capacity + entity_count == 0:
        return _empty_scene()
    arena = _new_arena(capacity + entity_count)
    written = 0
    for region, luts, stash in plan:
        px, py, pz = region.position.x, region.position.y, region.position.z
        for y0, y1, lx, ly, lz, picked in stash:
            written += _expand_slab_into(
                arena, written, y0, y1, lx, ly, lz, picked,
                luts, region, tier_visible, (px, py, pz),
            )
        stash.clear()
    if entity_rows is not None:
        written += _write_entity_rows(arena, written, entity_rows)
    x = arena.x[:written]
    y = arena.y[:written]
    z = arena.z[:written]
    x = x - int(x.min())
    y = y - int(y.min())
    z = z - int(z.min())
    c0, c1, r0, r1 = project_bounds(x, y, z, arena.vtx[:written], axes)
    span_w = int(math.ceil(c1 - c0 - _SPAN_SNAP)) + 1
    span_h = int(math.ceil(r1 - r0 - _SPAN_SNAP)) + 1
    return _Scene(
        x.astype(np.int32),
        y.astype(np.int32),
        z.astype(np.int32),
        arena.vtx[:written].copy(),
        arena.uv[:written].copy(),
        arena.ao[:written].copy(),
        arena.shade[:written].copy(),
        arena.dbl[:written].copy(),
        arena.texkey[:written].copy(),
        dict(key_arrays),
        span_w,
        span_h,
        axes,
    )


def _entity_rows(
    schem: Litematic,
    textures: Resolver,
    key_ids: dict[str, int],
    key_arrays: dict[str, np.ndarray],
) -> tuple | None:
    """Compile supported entities into scene-column row groups.

    Entity rows never occlude and carry no AO; anchors outside their
    region's block volume are skipped.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    vtxs: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    shades: list[np.ndarray] = []
    dbls: list[np.ndarray] = []
    texs: list[np.ndarray] = []
    for name in sorted(schem.regions):
        region = schem.regions[name]
        for ent in region.entities:
            stem = ent.id.rpartition(":")[2]
            cx, cy, cz = (
                math.floor(ent.pos[0]),
                math.floor(ent.pos[1]),
                math.floor(ent.pos[2]),
            )
            if not (
                0 <= cx < region.size.x
                and 0 <= cy < region.size.y
                and 0 <= cz < region.size.z
            ):
                continue
            props = entity_props(ent.data)
            quads = _entity_quads_for(ent, textures, props)
            if quads is None or len(quads) == 0:
                continue
            ids = _texkey_ids(quads, textures, key_ids, key_arrays)
            # Builders center x/z on the cell and anchor y at its floor, so
            # y takes the full fraction while x/z take the off-center part.
            # Frames bake their full wall placement via the facing turns
            # (pos is the cell center on every axis) and stay unshifted;
            # the z-buffer handles their wall-hung depth on its own.
            if stem == "item_frame" or stem == "glow_item_frame":
                fx = fy = fz = 0.0
            else:
                fx = (ent.pos[0] - cx - 0.5) * 16.0
                fy = (ent.pos[1] - cy) * 16.0
                fz = (ent.pos[2] - cz - 0.5) * 16.0
            vtx = quads["vtx"].copy()
            vtx[..., 0] += fx
            vtx[..., 1] += fy
            vtx[..., 2] += fz
            count = len(quads)
            xs.append(np.full(count, region.position.x + cx, np.int32))
            ys.append(np.full(count, region.position.y + cy, np.int32))
            zs.append(np.full(count, region.position.z + cz, np.int32))
            vtxs.append(vtx)
            uvs.append(quads["uv"])
            shades.append(quads["shade"])
            dbls.append(quads["dbl"])
            texs.append(ids)
    if not xs:
        return None
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(zs),
        np.concatenate(vtxs),
        np.concatenate(uvs),
        np.concatenate(shades),
        np.concatenate(dbls),
        np.concatenate(texs),
    )


def _write_entity_rows(arena: _Arena, offset: int, rows: tuple) -> int:
    """Append entity rows after the block pass; every tier stays renderable."""
    ex, ey, ez, evtx, euv, eshade, edbl, etex = rows
    count = len(ex)
    end = offset + count
    arena.x[offset:end] = ex
    arena.y[offset:end] = ey
    arena.z[offset:end] = ez
    arena.vtx[offset:end] = evtx
    arena.uv[offset:end] = euv
    arena.ao[offset:end] = 1.0
    arena.shade[offset:end] = eshade
    arena.dbl[offset:end] = edbl
    arena.texkey[offset:end] = etex
    return count


def _entity_quads_for(
    ent, textures: Resolver, props: dict[str, object]
) -> np.ndarray | None:
    """Dispatch one entity to its builder by id stem."""
    stem = ent.id.rpartition(":")[2]
    if stem == "item":
        item_id = _entity_item_id(ent.data)
        if item_id is None:
            return None
        return item_entity_quads(item_id, textures)
    if stem in ("item_frame", "glow_item_frame"):
        return item_frame_quads(props, _entity_item_id(ent.data), textures)
    if stem == "armor_stand":
        return armor_stand_quads(props, textures)
    return None


def _entity_item_id(data: Mapping) -> str | None:
    item = data.get("Item")
    if isinstance(item, Mapping):
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            return item_id
    return None


def _camera_row_counts(luts: _Luts, tier_visible: np.ndarray) -> np.ndarray:
    """Camera-facing pool-row count per palette entry under this view."""
    pool = luts.pool
    if pool is None:
        return np.zeros(len(luts.offsets) - 1, np.int64)
    counts = np.zeros(len(luts.offsets) - 1, np.int64)
    for i in range(len(counts)):
        rows = pool[luts.offsets[i] : luts.offsets[i + 1]]
        # Double-sided decals (plants, sprites) show from every direction.
        counts[i] = int((tier_visible[rows["shade"]] | rows["dbl"]).sum())
    return counts


def _empty_scene() -> _Scene:
    """Return the zero-quad scene with the v1 default canvas spans."""
    return _Scene(
        np.zeros(0, np.int32),
        np.zeros(0, np.int32),
        np.zeros(0, np.int32),
        np.zeros((0, 4, 3), np.float32),
        np.zeros((0, 4, 2), np.float32),
        np.zeros((0, 4), np.float32),
        np.zeros(0, np.uint8),
        np.zeros(0, bool),
        np.zeros(0, np.int32),
        {},
        2,
        4,
        view_axes(45.0, ISO_SLANT_DEG),
    )


def _cull_classes(palette) -> np.ndarray:
    """Pair class per palette entry; negative entries never pair.

    Same block name pairs (glass, ice, slime, water levels); the water
    family pairs across names; leaves never pair (cutout leaves default).
    """
    classes = np.full(len(palette), -1, np.int32)
    ids: dict[str, int] = {}
    for i, state in enumerate(palette):
        if state.name in _AIR_NAMES:
            continue
        if state.name.partition(":")[2].endswith("leaves"):
            continue
        if state.name in _WATER_FAMILY:
            classes[i] = 0
            continue
        key = state.name
        if key not in ids:
            ids[key] = len(ids) + 1
        classes[i] = ids[key]
    return classes


def _palette_luts(
    palette, textures: Resolver, key_ids: dict[str, int], key_arrays: dict
) -> _Luts:
    """Resolve every palette entry to quads, occl flags, and texkey ids."""
    count = len(palette)
    is_air = np.zeros(count, bool)
    occludes = np.zeros(count, bool)
    cull_class = _cull_classes(palette)
    quads_pieces: list[np.ndarray] = []
    id_pieces: list[np.ndarray] = []
    lengths = np.zeros(count + 1, np.int64)
    for i, state in enumerate(palette):
        if state.name in _AIR_NAMES:
            is_air[i] = True
            continue
        res = textures.resolve(state)
        occludes[i] = res.occludes
        quads_pieces.append(res.quads)
        id_pieces.append(_texkey_ids(res.quads, textures, key_ids, key_arrays))
        lengths[i + 1] = len(res.quads)
    if not quads_pieces:
        return _Luts(
            is_air, occludes, None, np.cumsum(lengths), np.zeros(0, np.int32),
            cull_class,
        )
    pool = (
        np.concatenate(quads_pieces) if len(quads_pieces) > 1 else quads_pieces[0]
    )
    ids = np.concatenate(id_pieces) if len(id_pieces) > 1 else id_pieces[0]
    return _Luts(is_air, occludes, pool, np.cumsum(lengths), ids, cull_class)


def _texkey_ids(
    quads: np.ndarray,
    textures: Resolver,
    key_ids: dict[str, int],
    key_arrays: dict,
) -> np.ndarray:
    """Map a quad array's texkeys to shared scene ids, registering arrays."""
    unique, inverse = np.unique(quads["texkey"], return_inverse=True)
    keys = [str(key) for key in unique]
    fresh = [key for key in keys if key not in key_ids]
    arrays = textures.texkey_arrays(fresh)
    for key in fresh:
        key_ids[key] = len(key_ids)
        key_arrays[key] = arrays[key]
    lut = np.array([key_ids[key] for key in keys], np.int32)
    return lut[inverse.reshape(-1)]


def _occlusion_window(
    widx: np.ndarray, occludes: np.ndarray, wy0: int
) -> tuple[np.ndarray, tuple[int, int, int], int]:
    """False-padded flat occlusion grid over the window's index rows.

    AO and cullface lookups reach at most one row beyond the slab, so the
    window (rows [wy0, wy1), unpacked on demand) never needs the rest of
    the region; region-edge rows fall to the False padding exactly like
    the vanilla out-of-bounds air.
    """
    sx, win_h, sz = widx.shape
    padded = np.zeros((sx + 2, win_h + 2, sz + 2), bool)
    padded[1 : sx + 1, 1 : win_h + 1, 1 : sz + 1] = occludes[widx]
    strides = ((win_h + 2) * (sz + 2), sz + 2, 1)
    return padded.ravel(), strides, wy0


def _visible_slabs(region, luts: _Luts):
    """Yield each slab's row span plus its visible cells for the arena.

    A cell is skipped only when all six neighbors occlude: every face it
    could emit is buried inside opaque geometry, whatever the camera.
    """
    sx, sy, sz = region.size
    is_air, occludes = luts.is_air, luts.occludes
    for y0 in range(0, sy, _Y_SLAB):
        y1 = min(y0 + _Y_SLAB, sy)
        body = region.slab_indices(y0, y1)
        present = ~is_air[body]
        occ_body = occludes[body]
        upper = occludes[body][:, 1:, :]
        if y1 == sy:
            next_row = np.zeros((sx, 1, sz), bool)
        else:
            # One extra row past the slab edge; a single-row unpack keeps
            # the visibility pass from paying for the next slab's body.
            next_row = occludes[region.slab_indices(y1, y1 + 1)]
        upper = np.concatenate([upper, next_row], axis=1)
        lower = occ_body[:, :-1, :]
        if y0 == 0:
            prev_row = np.zeros((sx, 1, sz), bool)
        else:
            prev_row = occludes[region.slab_indices(y0 - 1, y0)]
        lower = np.concatenate([prev_row, lower], axis=1)
        enclosed = upper & lower
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            enclosed = enclosed & _shifted(occ_body, dx, dz)
        visible = present & ~enclosed
        if not visible.any():
            continue
        lx, ly, lz = np.nonzero(visible)
        yield y0, y1, lx, ly + y0, lz, body[visible]


def _shifted(grid: np.ndarray, dx: int, dz: int) -> np.ndarray:
    """Neighbor occlusion one unit step along (dx, dz); outside is False."""
    out = np.zeros_like(grid)
    if dx > 0:
        out[:-1, :, :] = grid[1:, :, :]
    elif dx < 0:
        out[1:, :, :] = grid[:-1, :, :]
    elif dz > 0:
        out[:, :, :-1] = grid[:, :, 1:]
    elif dz < 0:
        out[:, :, 1:] = grid[:, :, :-1]
    return out


def _expand_slab_into(
    arena: _Arena,
    offset: int,
    y0: int,
    y1: int,
    lx: np.ndarray,
    ly: np.ndarray,
    lz: np.ndarray,
    picked: np.ndarray,
    luts: _Luts,
    region,
    tier_visible: np.ndarray,
    position: tuple[int, int, int],
) -> int:
    """Expand one stashed slab into the arena; return the quads written."""
    picked = picked.astype(np.int64)
    pool = luts.pool
    if pool is None:
        return 0
    counts = luts.offsets[picked + 1] - luts.offsets[picked]
    total = int(counts.sum())
    if total == 0:
        return 0
    cell_of = np.repeat(
        np.arange(len(picked), dtype=np.int32), counts
    )
    starts = np.cumsum(counts) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    pool_idx = np.repeat(luts.offsets[picked], counts) + within
    # Structured rows carry a 256-byte texkey, so gather fields and chunk.
    model_shade = pool["shade"][pool_idx]
    keep = tier_visible[model_shade] | pool["dbl"][pool_idx]
    kept_at = np.flatnonzero(keep)
    pool_rows = pool_idx[kept_at]
    del pool_idx, keep
    model_shade = model_shade[kept_at]
    cell_of = cell_of[kept_at]
    wy0 = max(y0 - 1, 0)
    widx = region.slab_indices(wy0, min(y1 + 1, region.size.y))
    occ = _occlusion_window(widx, luts.occludes, wy0)
    px, py, pz = position
    written = 0
    chunk = 1 << 20
    for start in range(0, len(pool_rows), chunk):
        stop = min(start + chunk, len(pool_rows))
        rows_c = pool_rows[start:stop]
        cell_of_c = cell_of[start:stop]
        cells = np.stack(
            [lx[cell_of_c], ly[cell_of_c], lz[cell_of_c]], axis=1
        )
        alive = _cull_alive(
            pool["cull"][rows_c],
            cells,
            picked[cell_of_c],
            luts.cull_class,
            occ,
            widx,
            wy0,
            region.size.y,
        )
        cells = cells[alive]
        rows = rows_c[alive]
        vtx = pool["vtx"][rows]
        dbl = pool["dbl"][rows]
        end = offset + written + len(cells)
        arena.x[offset + written : end] = cells[:, 0] + px
        arena.y[offset + written : end] = cells[:, 1] + py
        arena.z[offset + written : end] = cells[:, 2] + pz
        arena.vtx[offset + written : end] = vtx
        arena.uv[offset + written : end] = pool["uv"][rows]
        arena.ao[offset + written : end] = _vertex_ao(
            vtx, model_shade[start:stop][alive], dbl, cells, *occ
        )
        arena.shade[offset + written : end] = model_shade[start:stop][alive]
        arena.dbl[offset + written : end] = dbl
        arena.texkey[offset + written : end] = luts.texkey_ids[rows]
        written += len(cells)
    return written


def _cull_alive(
    cull: np.ndarray,
    cells: np.ndarray,
    cell_palette: np.ndarray,
    cull_class: np.ndarray,
    occ: tuple[np.ndarray, tuple[int, int, int], int],
    widx: np.ndarray,
    wy0: int,
    sy: int,
) -> np.ndarray:
    """Alive mask after neighbor culling; no-cullface rows always alive."""
    mask = np.ones(len(cull), bool)
    rows = np.nonzero(cull != CULL_NONE)[0]
    if len(rows) == 0:
        return mask
    nb = cells[rows] + _TIER_NORMALS[cull[rows].astype(np.int64)]
    sx, _, sz = widx.shape
    valid = (
        (nb[..., 0] >= 0) & (nb[..., 0] < sx)
        & (nb[..., 1] >= 0) & (nb[..., 1] < sy)
        & (nb[..., 2] >= 0) & (nb[..., 2] < sz)
    )
    nb_clipped = np.stack(
        [
            np.clip(nb[..., 0], 0, sx - 1),
            np.clip(nb[..., 1], 0, sy - 1),
            np.clip(nb[..., 2], 0, sz - 1),
        ],
        axis=1,
    )
    flat, (st_x, st_y, st_z), win_y0 = occ
    win_h = st_x // st_y - 2
    nb_pad = np.stack(
        [
            np.clip(nb[..., 0], -1, sx + 1),
            np.clip(nb[..., 1], wy0 - 1, wy0 + win_h),
            np.clip(nb[..., 2], -1, sz + 1),
        ],
        axis=1,
    )
    occluded = flat[
        (nb_pad[..., 0] + 1) * st_x
        + (nb_pad[..., 1] + 1 - win_y0) * st_y
        + (nb_pad[..., 2] + 1) * st_z
    ]
    neighbor_pi = widx[
        nb_clipped[..., 0],
        nb_clipped[..., 1] - wy0,
        nb_clipped[..., 2],
    ]
    own = cull_class[cell_palette[rows]]
    pair = (
        valid
        & (own >= 0)
        & (own == cull_class[neighbor_pi])
    )
    mask[rows] = ~(occluded | pair)
    return mask


def _vertex_ao(
    vtx: np.ndarray,
    shade: np.ndarray,
    dbl: np.ndarray,
    cells: np.ndarray,
    flat: np.ndarray,
    strides: tuple[int, int, int],
    y_base: int = 0,
) -> np.ndarray:
    """Per-vertex AO multipliers from the padded flat grid, world space."""
    tier = shade.astype(np.int32)
    axes = _TIER_AXES[tier]
    normals = _TIER_NORMALS[tier]
    st_x, st_y, st_z = strides
    steps = np.array([st_x, st_y, st_z], np.int32)
    t1 = steps[axes[:, 1]]
    t2 = steps[axes[:, 2]]
    pick1 = axes[:, 1][:, None, None]
    pick2 = axes[:, 2][:, None, None]
    # A vertex exactly on the block center plane (coord 8.0) signs to 0 and
    # samples the straight-out neighbor; the shape table keeps coords off 8.
    s1 = np.sign(np.take_along_axis(vtx, pick1, axis=2)[:, :, 0] - 8.0)
    s2 = np.sign(np.take_along_axis(vtx, pick2, axis=2)[:, :, 0] - 8.0)
    s1 = s1.astype(np.int32)
    s2 = s2.astype(np.int32)
    cells32 = cells.astype(np.int32)
    base = (
        (cells32[:, 0] + 1) * np.int32(st_x)
        + (cells32[:, 1] + 1 - y_base) * np.int32(st_y)
        + (cells32[:, 2] + 1) * np.int32(st_z)
        + normals[:, 0] * np.int32(st_x)
        + normals[:, 1] * np.int32(st_y)
        + normals[:, 2] * np.int32(st_z)
    )[:, None]
    side1 = flat[base + s1 * t1[:, None]].astype(np.int32)
    side2 = flat[base + s2 * t2[:, None]].astype(np.int32)
    corner = flat[base + s1 * t1[:, None] + s2 * t2[:, None]].astype(np.int32)
    level = np.where(side1 & side2, 0, 3 - (side1 + side2 + corner))
    ao = np.asarray(AO_LEVELS, np.float32)[level]
    ao[dbl] = 1.0
    return ao
