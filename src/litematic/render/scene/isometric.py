"""Scene composition: quad expansion, yaw remap, camera filtering, AO."""

from typing import Mapping, NamedTuple

import numpy as np

from litematic.parse.model import Litematic
from litematic.render.errors import RenderError
from litematic.render.pack import Textures
from litematic.render.pack.texture import _tint_for
from litematic.render.quads import (
    AO_LEVELS,
    CULL_NONE,
    TIER_DOWN,
    TIER_EAST,
    TIER_NORTH,
    TIER_SOUTH,
    TIER_UP,
    TIER_WEST,
    unit_cube_quads,
)
from litematic.render.shapes import builtin_quads

_AIR_NAMES = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
_WATER_FAMILY = {"minecraft:water", "minecraft:bubble_column"}
_Y_SLAB = 64
_MAGENTA_TEXKEY = "__magenta__"
_MAGENTA_ARRAY = np.full((16, 16, 4), (255, 0, 255, 255), np.uint8)
_MAGENTA_ARRAY.flags.writeable = False  # painter must never mutate in place
_CAMERA_TIERS = (TIER_UP, TIER_SOUTH, TIER_EAST)

# Face normal vectors per tier in block space; AO samples sit one step out
# along the normal and one step along each tangent sign.
_TIER_NORMALS = np.zeros((6, 3), np.int64)
_TIER_NORMALS[TIER_DOWN] = (0, -1, 0)
_TIER_NORMALS[TIER_UP] = (0, 1, 0)
_TIER_NORMALS[TIER_NORTH] = (0, 0, -1)
_TIER_NORMALS[TIER_SOUTH] = (0, 0, 1)
_TIER_NORMALS[TIER_WEST] = (-1, 0, 0)
_TIER_NORMALS[TIER_EAST] = (1, 0, 0)
_TIER_OF_NORMAL = {tuple(row): tier for tier, row in enumerate(_TIER_NORMALS)}

# Per tier: normal axis then the two tangent axes for AO corner sampling.
_TIER_AXES = np.zeros((6, 3), np.int64)
_TIER_AXES[TIER_DOWN] = (1, 0, 2)
_TIER_AXES[TIER_UP] = (1, 0, 2)
_TIER_AXES[TIER_NORTH] = (2, 0, 1)
_TIER_AXES[TIER_SOUTH] = (2, 0, 1)
_TIER_AXES[TIER_WEST] = (0, 1, 2)
_TIER_AXES[TIER_EAST] = (0, 1, 2)


class _Scene(NamedTuple):
    """Camera-facing quads after yaw rotation and vertex AO, scene coords."""

    x: np.ndarray  # int32 per-quad block x
    y: np.ndarray  # int32 per-quad block y
    z: np.ndarray  # int32 per-quad block z
    quad_vtx: np.ndarray  # (Q, 4, 3) f32 block-local, yaw rotated
    quad_uv: np.ndarray  # (Q, 4, 2) f32 texel coords, never rotated
    quad_ao: np.ndarray  # (Q, 4) f32 per-vertex AO multipliers
    quad_shade: np.ndarray  # (Q,) u1 tier, limited to up/south/east
    quad_dbl: np.ndarray  # (Q,) bool double-sided flag
    texkey_ids: np.ndarray  # (Q,) i32 index into texkeys
    texkeys: dict[str, np.ndarray]  # unique texkey -> (16, 16, 4) array
    span_w: int  # canvas width in pixels = 2t * span_w at scale t
    span_h: int  # canvas height in pixels = t * span_h at scale t


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


def _yaw_tier_map(yaw: int) -> np.ndarray:
    """Remap the six face tiers through the yaw rotation formula."""
    out = np.empty(6, np.uint8)
    for tier in range(6):
        nx, ny, nz = (int(v) for v in _TIER_NORMALS[tier])
        if yaw == 1:
            rx, rz = nz, -nx
        elif yaw == 2:
            rx, rz = -nx, -nz
        elif yaw == 3:
            rx, rz = -nz, nx
        else:
            rx, rz = nx, nz
        out[tier] = _TIER_OF_NORMAL[(rx, ny, rz)]
    return out


_YAW_TIER_MAPS = tuple(_yaw_tier_map(yaw) for yaw in range(4))


def _horizontal_cull_dirs(
    yaw: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """World-space neighbor steps whose faces face the camera after yaw."""
    dirs: list[tuple[int, int]] = []
    for tier in (TIER_EAST, TIER_SOUTH):
        nx, _, nz = (int(v) for v in _TIER_NORMALS[tier])
        for _ in range((4 - yaw) % 4):
            nx, nz = nz, -nx
        dirs.append((nx, nz))
    return dirs[0], dirs[1]


class _Luts(NamedTuple):
    """Per-palette-entry lookups for culling and quad expansion."""

    is_air: np.ndarray  # bool
    occludes: np.ndarray  # bool
    pool: np.ndarray | None  # (P,) QUAD_DTYPE, None when every entry is air
    offsets: np.ndarray  # (E + 1,) i8 start of each entry's rows in pool
    texkey_ids: np.ndarray  # (P,) i32 per-pool-row texkey id
    cull_class: np.ndarray  # (P,) i32 pair class, negative entries never pair


class _Piece(NamedTuple):
    """One slab's expanded camera-facing quads in world coordinates."""

    cells: np.ndarray  # (K, 3) i8 world block cell per quad
    vtx: np.ndarray  # (K, 4, 3) f32 block-local, yaw rotated
    uv: np.ndarray  # (K, 4, 2) f32
    ao: np.ndarray  # (K, 4) f32
    shade: np.ndarray  # (K,) u1 camera-facing tier
    dbl: np.ndarray  # (K,) bool
    texkey_ids: np.ndarray  # (K,) i32


def _compose(schem: Litematic, yaw: int, textures: Textures) -> _Scene:
    """Collect visible blocks from all regions into a quad scene."""
    try:
        return _compose_inner(schem, yaw, textures)
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError(f"compose: scene composition failed: {exc}") from exc


def _compose_inner(schem: Litematic, yaw: int, textures: Textures) -> _Scene:
    """Expand every region's visible cells into yawed, filtered quads."""
    key_ids: dict[str, int] = {}
    key_arrays: dict[str, np.ndarray] = {}
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    pieces: list[_Piece] = []
    tier_map = _YAW_TIER_MAPS[yaw]
    dir_a, dir_b = _horizontal_cull_dirs(yaw)
    for name in sorted(schem.regions):
        region = schem.regions[name]
        luts = _palette_luts(region.palette, textures, key_ids, key_arrays)
        if luts.pool is None:
            continue
        occ = _occlusion_flat(region, luts.occludes)
        idx = region.block_indices
        px, py, pz = region.position.x, region.position.y, region.position.z
        for cx, cy, cz, picked in _visible_slabs(
            region, luts, dir_a, dir_b
        ):
            piece = _expand_slab(
                cx, cy, cz, picked, luts, occ, idx, tier_map, yaw
            )
            if len(piece.cells) == 0:
                continue
            pieces.append(piece)
            xs.append(piece.cells[:, 0] + px)
            ys.append(piece.cells[:, 1] + py)
            zs.append(piece.cells[:, 2] + pz)
    if not xs:
        return _empty_scene()
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    x -= int(x.min())
    y -= int(y.min())
    z -= int(z.min())
    rx, rz = _yaw_rot(x, z, yaw, int(x.max()) + 1, int(z.max()) + 1)
    span_w = int(rx.max()) + int(rz.max()) + 2
    span_h = int(rx.max()) + int(rz.max()) + 2 * int(y.max()) + 4
    return _Scene(
        rx.astype(np.int32),
        y.astype(np.int32),
        rz.astype(np.int32),
        np.concatenate([piece.vtx for piece in pieces]),
        np.concatenate([piece.uv for piece in pieces]),
        np.concatenate([piece.ao for piece in pieces]),
        np.concatenate([piece.shade for piece in pieces]),
        np.concatenate([piece.dbl for piece in pieces]),
        np.concatenate([piece.texkey_ids for piece in pieces]).astype(np.int32),
        dict(key_arrays),
        span_w,
        span_h,
    )


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
    palette, textures: Textures, key_ids: dict[str, int], key_arrays: dict
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
        occludes[i] = textures.occludes(state.name, state.properties)
        quads = _state_quads(state, textures)
        quads_pieces.append(quads)
        id_pieces.append(_texkey_ids(quads, textures, key_ids, key_arrays))
        lengths[i + 1] = len(quads)
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


def _state_quads(state, textures: Textures) -> np.ndarray:
    """Resolve one state: pack quads, builtin shapes, then magenta cube."""
    quads = textures.quads(state.name, state.properties)
    if quads is not None:
        return quads
    variants: tuple[Mapping[str, str], ...] = (state.properties, {})
    for props in variants:
        built = builtin_quads(state.name, props)
        if built is not None:
            if _backfill_builtin(state.name, built, textures.arrays):
                return built
            break
    return unit_cube_quads(_MAGENTA_TEXKEY)


# Vanilla base-material texture conventions for model-less packs (v2.1
# extension point, hand-maintained): family suffix stripped to the base
# material, woods to their planks, dye carpets to wool, plus the vanilla
# plural form as a second candidate. Longest suffixes must come first.
_FAMILY_SUFFIXES: tuple[str, ...] = (
    "_pressure_plate",
    "_fence_gate",
    "_wall_banner",
    "_wall_sign",
    "_trapdoor",
    "_button",
    "_stairs",
    "_fence",
    "_carpet",
    "_slab",
    "_wall",
    "_sign",
    "_banner",
    "_door",
)
_WOODS = frozenset(
    {
        "oak",
        "spruce",
        "birch",
        "jungle",
        "acacia",
        "dark_oak",
        "mangrove",
        "cherry",
        "bamboo",
        "crimson",
        "warped",
    }
)
_DYES = frozenset(
    {
        "white",
        "orange",
        "magenta",
        "light_blue",
        "yellow",
        "lime",
        "pink",
        "gray",
        "light_gray",
        "cyan",
        "purple",
        "blue",
        "brown",
        "green",
        "red",
        "black",
    }
)


def _backfill_builtin(
    name: str, quads: np.ndarray, arrays: dict[str, np.ndarray]
) -> bool:
    """Fill builtin quads' texkeys from direct, then material, keys.

    Builders may emit final per-quad texkeys (wire power tints); those
    pass through when every key resolves in the arrays. Direct matches
    (same-name plus tint variant, extraction-registered) always win; the
    vanilla naming conventions fire only when no direct texture exists,
    and unresolved families fall through to magenta.
    """
    if np.all(quads["texkey"] != ""):
        return all(str(key) in arrays for key in quads["texkey"])
    ns, _, stem = name.partition(":")
    prefix = "" if ns in ("", "minecraft") else f"{ns}:"
    base = f"{prefix}block/{stem or name}"
    tint = _tint_for(name, "")
    direct = [base]
    if tint is not None:
        direct.insert(0, f"{base}@{tint[0]},{tint[1]},{tint[2]}")
    for key in direct:
        if key in arrays:
            quads["texkey"] = key
            return True
    door = _door_texkeys(prefix, stem)
    if door is not None and all(key in arrays for key in door):
        quads["texkey"] = np.where(
            quads["vtx"][..., 1].mean(axis=1) >= 8.0, door[0], door[1]
        )
        return True
    for key in _material_keys(prefix, stem):
        if key in arrays:
            quads["texkey"] = key
            return True
    return False


def _door_texkeys(prefix: str, stem: str) -> tuple[str, str] | None:
    """(top, bottom) texture keys for a vanilla door stem, else None."""
    if not stem.endswith("_door"):
        return None
    material = stem[: -len("_door")]
    return (
        f"{prefix}block/{material}_door_top",
        f"{prefix}block/{material}_door_bottom",
    )


def _material_keys(prefix: str, stem: str) -> tuple[str, ...]:
    """Ordered base-material texture keys per vanilla naming rules."""
    if stem.endswith("_pane"):
        # Vanilla pane models texture from <stem>_top; the bare glass is
        # the second chance.
        material = stem[: -len("_pane")]
        return (
            f"{prefix}block/{stem}_top",
            f"{prefix}block/{material}",
            f"{prefix}block/{material}s",
        )
    for suffix in _FAMILY_SUFFIXES:
        if stem.endswith(suffix):
            material = stem[: -len(suffix)]
            break
    else:
        return ()
    if material in _WOODS:
        return (f"{prefix}block/{material}_planks",)
    if suffix == "_carpet" and material in _DYES:
        return (f"{prefix}block/{material}_wool",)
    return (f"{prefix}block/{material}", f"{prefix}block/{material}s")


def _state_is_magenta(state, textures: Textures) -> bool:
    """True when compose resolves this state to the magenta fallback cube."""
    quads = _state_quads(state, textures)
    return bool(np.any(quads["texkey"] == _MAGENTA_TEXKEY))


def _texkey_ids(
    quads: np.ndarray,
    textures: Textures,
    key_ids: dict[str, int],
    key_arrays: dict,
) -> np.ndarray:
    """Map a quad array's texkeys to shared scene ids, registering arrays."""
    unique, inverse = np.unique(quads["texkey"], return_inverse=True)
    lut = np.array(
        [_register_key(str(key), textures, key_ids, key_arrays) for key in unique],
        np.int32,
    )
    return lut[inverse.reshape(-1)]


def _register_key(
    key: str,
    textures: Textures,
    key_ids: dict[str, int],
    key_arrays: dict,
) -> int:
    """Assign a stable id and texture array to a texkey on first sight."""
    if key in key_ids:
        return key_ids[key]
    if key == _MAGENTA_TEXKEY:
        array = _MAGENTA_ARRAY
    elif key in textures.arrays:
        array = textures.arrays[key]
    else:
        raise RenderError(f"compose: quad texkey missing from pack arrays: {key}")
    key_ids[key] = len(key_ids)
    key_arrays[key] = array
    return key_ids[key]


def _occlusion_flat(
    region, occludes: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Region occlusion as a False-padded flat grid plus per-axis strides."""
    idx = region.block_indices
    sx, sy, sz = idx.shape
    padded = np.zeros((sx + 2, sy + 2, sz + 2), bool)
    for y0 in range(0, sy, _Y_SLAB):
        y1 = min(y0 + _Y_SLAB, sy)
        padded[1 : sx + 1, 1 + y0 : 1 + y1, 1 : sz + 1] = occludes[
            idx[:, y0:y1, :]
        ]
    return padded.ravel(), ((sy + 2) * (sz + 2), sz + 2, 1)


def _visible_slabs(
    region,
    luts: _Luts,
    dir_a: tuple[int, int],
    dir_b: tuple[int, int],
):
    """Yield (x, y, z, palette index) arrays for each slab's visible cells."""
    idx = region.block_indices
    sx, sy, sz = idx.shape
    is_air, occludes = luts.is_air, luts.occludes
    for y0 in range(0, sy, _Y_SLAB):
        y1 = min(y0 + _Y_SLAB, sy)
        body = idx[:, y0:y1, :]
        present = ~is_air[body]
        occ_body = occludes[body]
        rows = idx[:, y0 : min(y1 + 1, sy), :]
        upper = occludes[rows][:, 1:, :]
        if y1 == sy:
            upper = np.concatenate([upper, np.zeros((sx, 1, sz), bool)], axis=1)
        occ_a = _shifted(occ_body, *dir_a)
        occ_b = _shifted(occ_body, *dir_b)
        visible = present & ~(upper & occ_a & occ_b)
        if not visible.any():
            continue
        lx, ly, lz = np.nonzero(visible)
        yield lx, ly + y0, lz, body[visible]


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


def _expand_slab(
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    picked: np.ndarray,
    luts: _Luts,
    occ: tuple[np.ndarray, tuple[int, int, int]],
    idx: np.ndarray,
    tier_map: np.ndarray,
    yaw: int,
) -> _Piece:
    """Expand one slab's cells into filtered world quads with vertex AO."""
    picked = picked.astype(np.int64)
    pool = luts.pool
    if pool is None:
        return _empty_piece()
    counts = luts.offsets[picked + 1] - luts.offsets[picked]
    total = int(counts.sum())
    if total == 0:
        return _empty_piece()
    cell_of = np.repeat(np.arange(len(picked), dtype=np.int64), counts)
    starts = np.cumsum(counts) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    pool_idx = np.repeat(luts.offsets[picked], counts) + within
    quads = pool[pool_idx]
    cells = np.stack([cx[cell_of], cy[cell_of], cz[cell_of]], axis=1)
    ids = luts.texkey_ids[pool_idx]
    new_tier = tier_map[quads["shade"]]
    keep = np.isin(new_tier, _CAMERA_TIERS)
    quads = quads[keep]
    cells = cells[keep]
    ids = ids[keep]
    cell_of = cell_of[keep]
    alive = _cull_alive(
        quads, cells, picked[cell_of], luts.cull_class, occ, idx
    )
    quads = quads[alive]
    return _Piece(
        cells[alive],
        _rotate_vertices(quads["vtx"], yaw),
        quads["uv"],
        _vertex_ao(quads, cells[alive], *occ),
        new_tier[keep][alive],
        quads["dbl"],
        ids[alive],
    )


def _cull_alive(
    quads: np.ndarray,
    cells: np.ndarray,
    cell_palette: np.ndarray,
    cull_class: np.ndarray,
    occ: tuple[np.ndarray, tuple[int, int, int]],
    idx: np.ndarray,
) -> np.ndarray:
    """Alive mask after neighbor culling; no-cullface rows always alive."""
    mask = np.ones(len(quads), bool)
    cull = quads["cull"]
    rows = np.nonzero(cull != CULL_NONE)[0]
    if len(rows) == 0:
        return mask
    nb = cells[rows] + _TIER_NORMALS[cull[rows].astype(np.int64)]
    sx, sy, sz = idx.shape
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
    flat, (st_x, st_y, st_z) = occ
    nb_pad = np.stack(
        [
            np.clip(nb[..., 0], -1, sx + 1),
            np.clip(nb[..., 1], -1, sy + 1),
            np.clip(nb[..., 2], -1, sz + 1),
        ],
        axis=1,
    )
    occluded = flat[
        (nb_pad[..., 0] + 1) * st_x
        + (nb_pad[..., 1] + 1) * st_y
        + (nb_pad[..., 2] + 1) * st_z
    ]
    neighbor_pi = idx[nb_clipped[..., 0], nb_clipped[..., 1], nb_clipped[..., 2]]
    own = cull_class[cell_palette[rows]]
    pair = (
        valid
        & (own >= 0)
        & (own == cull_class[neighbor_pi])
    )
    mask[rows] = ~(occluded | pair)
    return mask


def _empty_piece() -> _Piece:
    """Return a zero-quad slab piece."""
    return _Piece(
        np.zeros((0, 3), np.int64),
        np.zeros((0, 4, 3), np.float32),
        np.zeros((0, 4, 2), np.float32),
        np.zeros((0, 4), np.float32),
        np.zeros(0, np.uint8),
        np.zeros(0, bool),
        np.zeros(0, np.int32),
    )


def _rotate_vertices(vtx: np.ndarray, yaw: int) -> np.ndarray:
    """Rotate block-local vertices by the yaw formula, uv left untouched."""
    out = vtx
    for _ in range(yaw % 4):
        x, y, z = out[..., 0], out[..., 1], out[..., 2]
        out = np.stack([z, y, 16.0 - x], axis=-1)
    return np.asarray(out, dtype=np.float32)


def _vertex_ao(
    quads: np.ndarray,
    cells: np.ndarray,
    flat: np.ndarray,
    strides: tuple[int, int, int],
) -> np.ndarray:
    """Per-vertex AO multipliers from the padded flat occlusion grid."""
    tier = quads["shade"].astype(np.int64)
    axes = _TIER_AXES[tier]
    normals = _TIER_NORMALS[tier]
    st_x, st_y, st_z = strides
    steps = np.array([st_x, st_y, st_z], np.int64)
    t1 = steps[axes[:, 1]]
    t2 = steps[axes[:, 2]]
    vtx = quads["vtx"]
    pick1 = axes[:, 1][:, None, None]
    pick2 = axes[:, 2][:, None, None]
    # A vertex exactly on the block center plane (coord 8.0) signs to 0 and
    # samples the straight-out neighbor; the shape table keeps coords off 8.
    s1 = np.sign(np.take_along_axis(vtx, pick1, axis=2)[:, :, 0] - 8.0)
    s2 = np.sign(np.take_along_axis(vtx, pick2, axis=2)[:, :, 0] - 8.0)
    s1 = s1.astype(np.int64)
    s2 = s2.astype(np.int64)
    cells32 = cells.astype(np.int32)
    base = (
        (cells32[:, 0] + 1) * np.int32(st_x)
        + (cells32[:, 1] + 1) * np.int32(st_y)
        + (cells32[:, 2] + 1) * np.int32(st_z)
        + normals[:, 0] * np.int32(st_x)
        + normals[:, 1] * np.int32(st_y)
        + normals[:, 2] * np.int32(st_z)
    )[:, None]
    side1 = flat[base + s1 * t1[:, None]].astype(np.int64)
    side2 = flat[base + s2 * t2[:, None]].astype(np.int64)
    corner = flat[base + s1 * t1[:, None] + s2 * t2[:, None]].astype(np.int64)
    level = np.where(side1 & side2, 0, 3 - (side1 + side2 + corner))
    ao = np.asarray(AO_LEVELS, np.float32)[level]
    ao[quads["dbl"]] = 1.0
    return ao
