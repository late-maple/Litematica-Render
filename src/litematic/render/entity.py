"""Entities compiled to quads: dropped items, item frames, armor stands."""

from collections.abc import Mapping

import numpy as np

from litematic.render.pack.resolve import _MAGENTA_TEXKEY, FRAME_PAD_KEY
from litematic.render.quads import CULL_NONE, QUAD_DTYPE, TIER_SOUTH

# Vanilla ground display, frozen at the bob mid-point: generated items
# scale 0.5 (half-size sprite), block items 0.25 (mini cube), both hover
# 0.1 + 1/16 - bbox.minY above the floor (ItemEntityRenderer static pose).
ITEM_FLAT_SIZE = 8.0
ITEM_BLOCK_SCALE = 0.25
ITEM_MINI_BASE = 2.6

# Facing yaw turns matching the compose convention. Vanilla puts the pad
# on the wall OPPOSITE the facing (createBoundingBox: center.relative(
# facing, -0.46875)), the front looking along the facing.
_FACING_TURNS = {"south": 2, "north": 0, "east": 3, "west": 1}
# Vanilla Facing NBT byte -> facing name (up/down unsupported in v1).
_NBT_FACING = {0: "down", 1: "up", 2: "north", 3: "south", 4: "west", 5: "east"}
_ARMORSTAND_TEXKEY = "entity/armorstand/armorstand"

# Face tiers after one 90-degree turn (south -> east), matching compose.
_TIER_TURNS = {
    1: np.array([0, 1, 4, 5, 3, 2], np.uint8),
    2: np.array([0, 1, 3, 2, 5, 4], np.uint8),
    3: np.array([0, 1, 5, 4, 2, 3], np.uint8),
}



def item_entity_quads(item_id: str, textures) -> np.ndarray | None:
    """Cell-local quads for one dropped item, or None when unrenderable."""
    entry = _item_entry(item_id, textures)
    if entry is None:
        return None
    if "block" in entry:
        return _mini_block_quads(entry["block"], textures)
    return _flat_item_quads(entry["flat"], textures)



def magenta_entity_quads() -> np.ndarray:
    """Fallback sprite for unresolvable frame content, in the frame geometry."""
    return _one_quad(_MAGENTA_TEXKEY, _frame_corners())



def entity_props(data: object) -> dict[str, object]:
    """Normalize raw entity NBT into the properties builders consume."""
    src: Mapping[str, object] = data if isinstance(data, Mapping) else {}
    key = src.get("Facing")
    props: dict[str, object] = {
        "facing": _NBT_FACING.get(key) if isinstance(key, int) else None
    }
    rotation = src.get("Rotation")
    if isinstance(rotation, (list, tuple)) and rotation:
        yaw = _as_float(rotation[0])
        if yaw is not None:
            props["yaw"] = yaw
    for flag in ("Invisible", "Small", "NoBasePlate", "ShowArms"):
        if flag in src:
            props[flag] = bool(src[flag])
    turns = _as_int(src.get("ItemRotation", 0))
    props["item_rotation"] = 0 if turns is None else turns
    return props


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_int(value: object) -> int | None:
    number = _as_float(value)
    return None if number is None else int(number)



def item_frame_quads(
    properties: dict[str, object],
    item_id: str | None,
    textures,
) -> np.ndarray | None:
    """Wall-frame pad plus content quads, or None when nothing renders."""
    facing = properties.get("facing")
    if facing not in _FACING_TURNS:
        return None
    turns = _FACING_TURNS[facing]
    pad = (
        None
        if properties.get("Invisible")
        else _frame_pad_quads(facing, textures)
    )
    content = _frame_content_quads(properties, item_id, textures)
    if content is not None:
        content = _turn_cell_quads(content, turns)
    if pad is None and content is None:
        return None
    return np.concatenate([q for q in (pad, content) if q is not None])



def armor_stand_quads(properties: dict[str, object], textures) -> np.ndarray | None:
    """Box-group quads for one armor stand, or None when unrenderable."""
    if properties.get("Invisible"):
        return None
    if _ARMORSTAND_TEXKEY not in textures.arrays:
        return None
    small = bool(properties.get("Small"))
    groups = [
        _part_quads(uv, box, size, pivot, flag, small)
        for name, uv, box, size, pivot, flag in _ARMOR_STAND_PARTS
        if not (flag == "arms" and not properties.get("ShowArms"))
        and not (flag == "plate" and properties.get("NoBasePlate"))
    ]
    quads = np.concatenate(groups)
    yaw = _as_float(properties.get("yaw"))
    if yaw:
        angle = np.deg2rad(yaw)
        cos, sin = np.float32(np.cos(angle)), np.float32(np.sin(angle))
        vtx = quads["vtx"]
        dx, dz = vtx[..., 0] - 8.0, vtx[..., 2] - 8.0
        vtx[..., 0] = 8.0 + dx * cos - dz * sin
        vtx[..., 2] = 8.0 + dx * sin + dz * cos
    return quads



def _item_entry(item_id: str, textures) -> dict[str, str] | None:
    """Display entry for an item, or a direct sprite when the pack has one."""
    entry = textures.item_display.get(item_id)
    if entry is not None:
        return entry
    direct = f"item/{item_id.rpartition(':')[2]}"
    if direct in textures.arrays:
        return {"flat": direct}
    return None



def _mini_block_quads(block_id: str, textures) -> np.ndarray | None:
    quads = textures.pack_quads(block_id)
    if quads is None or len(quads) == 0:
        return None
    out = quads.copy()
    vtx = out["vtx"]
    vtx[..., 0] = 8.0 - 8.0 * ITEM_BLOCK_SCALE + vtx[..., 0] * ITEM_BLOCK_SCALE
    vtx[..., 1] = ITEM_MINI_BASE + vtx[..., 1] * ITEM_BLOCK_SCALE
    vtx[..., 2] = 8.0 - 8.0 * ITEM_BLOCK_SCALE + vtx[..., 2] * ITEM_BLOCK_SCALE
    out["dbl"] = True
    out["cull"] = CULL_NONE
    return out



def _flat_item_quads(texkey: str, textures) -> np.ndarray | None:
    if texkey != _MAGENTA_TEXKEY and texkey not in textures.arrays:
        return None
    return _one_quad(texkey, _flat_corners())



def _flat_corners() -> np.ndarray:
    half = ITEM_FLAT_SIZE / 2
    top = ITEM_MINI_BASE + ITEM_FLAT_SIZE
    return np.array(
        [
            [8.0 - half, top, 8.0],
            [8.0 + half, top, 8.0],
            [8.0 + half, ITEM_MINI_BASE, 8.0],
            [8.0 - half, ITEM_MINI_BASE, 8.0],
        ],
        np.float32,
    )



def _one_quad(texkey: str, corners: np.ndarray) -> np.ndarray:
    quads = np.zeros(1, QUAD_DTYPE)
    quads["vtx"][0] = corners
    quads["uv"][0] = ((0, 0), (16, 0), (16, 16), (0, 16))
    quads["texkey"][0] = texkey
    quads["shade"][0] = TIER_SOUTH
    quads["dbl"][0] = True
    quads["cull"][0] = CULL_NONE
    return quads



def _frame_pad_quads(facing: str, textures) -> np.ndarray | None:
    pad = textures.mquad_map.get(FRAME_PAD_KEY)
    if pad is not None and len(pad):
        return _turn_cell_quads(pad, _FACING_TURNS[facing])
    # Texture-only packs: the frame sprite stands in for the missing model.
    if "item/item_frame" not in textures.arrays:
        return None
    quad = _one_quad(
        "item/item_frame",
        np.array(
            [
                [2.0, 14.0, 15.5],
                [14.0, 14.0, 15.5],
                [14.0, 2.0, 15.5],
                [2.0, 2.0, 15.5],
            ],
            np.float32,
        ),
    )
    quad["uv"][0] = ((2, 2), (14, 2), (14, 14), (2, 14))
    return _turn_cell_quads(quad, _FACING_TURNS[facing])



def item_content_texkey(item_id: str, textures) -> str | None:
    """Flat texkey drawn for an item, or None when nothing can render it."""
    entry = _item_entry(item_id, textures)
    if entry is None:
        return None
    if "flat" in entry:
        return entry["flat"] if entry["flat"] in textures.arrays else None
    plate = textures.pack_quads(entry["block"])
    if plate is None or len(plate) == 0:
        return None
    return str(plate["texkey"][0])



def _frame_content_quads(
    properties: dict[str, object],
    item_id: str | None,
    textures,
) -> np.ndarray | None:
    """Half-size sprite centered on the pad, flush with its front plane."""
    if item_id is None:
        return None
    texkey = item_content_texkey(item_id, textures)
    sprite = (
        magenta_entity_quads()
        if texkey is None
        else _one_quad(texkey, _frame_corners())
    )
    return _turn_content(sprite, _as_int(properties.get("item_rotation", 0)) or 0)



def _frame_corners() -> np.ndarray:
    """Content plane per vanilla FIXED: sprite mid at the pad front (z=15),
    half-size sprite centered on the pad center."""
    half = ITEM_FLAT_SIZE / 2
    return np.array(
        [
            [8.0 - half, 8.0 + half, 15.0],
            [8.0 + half, 8.0 + half, 15.0],
            [8.0 + half, 8.0 - half, 15.0],
            [8.0 - half, 8.0 - half, 15.0],
        ],
        np.float32,
    )



def _turn_content(sprite: np.ndarray, turns: int) -> np.ndarray:
    """Spin the pad-plane sprite in 45-degree item steps about the center."""
    out = sprite.copy()
    if turns:
        angle = np.deg2rad(45.0 * turns)
        cos, sin = np.float32(np.cos(angle)), np.float32(np.sin(angle))
        vtx = out["vtx"]
        dx, dy = vtx[..., 0] - 8.0, vtx[..., 1] - 8.0
        vtx[..., 0] = 8.0 + dx * cos - dy * sin
        vtx[..., 1] = 8.0 + dx * sin + dy * cos
    out["cull"] = CULL_NONE
    out["dbl"] = True
    return out



def _turn_cell_quads(quads: np.ndarray, turns: int) -> np.ndarray:
    """Rotate cell-local quads about the Y axis, compose turn convention."""
    turns %= 4
    if turns == 0:
        return quads
    out = quads.copy()
    vtx = out["vtx"]
    dx, dz = vtx[..., 0] - 8.0, vtx[..., 2] - 8.0
    if turns == 1:
        nx, nz = dz, -dx
    elif turns == 2:
        nx, nz = -dx, -dz
    else:
        nx, nz = -dz, dx
    vtx[..., 0] = 8.0 + nx
    vtx[..., 2] = 8.0 + nz
    out["shade"] = _TIER_TURNS[turns][out["shade"]]
    return out



# Vanilla ArmorStandModel boxes (64x64 texture space, model y-down, 24 =
# ground): (uv, box offset, size, pivot, flag). World mapping: x/z as-is,
# y = 24 - model_y; small applies the BabyModelTransform pivot/box scaling.
_ARMOR_STAND_PARTS = (
    ("head", (0, 0), (-1, -7, -1), (2, 7, 2), (0, 1, 0), "head"),
    ("body", (0, 26), (-6, 0, -1.5), (12, 3, 3), (0, 0, 0), ""),
    ("right_body_stick", (16, 0), (-3, 3, -1), (2, 7, 2), (0, 0, 0), ""),
    ("left_body_stick", (48, 16), (1, 3, -1), (2, 7, 2), (0, 0, 0), ""),
    ("shoulder_stick", (0, 48), (-4, 10, -1), (8, 2, 2), (0, 0, 0), ""),
    ("right_arm", (24, 0), (-2, -2, -1), (2, 12, 2), (-5, 2, 0), "arms"),
    ("left_arm", (32, 16), (0, -2, -1), (2, 12, 2), (5, 2, 0), "arms"),
    ("right_leg", (8, 0), (-1, 0, -1), (2, 11, 2), (-1.9, 12, 0), ""),
    ("left_leg", (40, 16), (-1, 0, -1), (2, 11, 2), (1.9, 12, 0), ""),
    ("base_plate", (0, 32), (-6, 11, -6), (12, 1, 12), (0, 12, 0), "plate"),
)
_PART_FACES = (
    ("down", 0),
    ("up", 1),
    ("north", 2),
    ("south", 3),
    ("west", 4),
    ("east", 5),
)



def _part_quads(
    uv: tuple[int, int],
    box: tuple[float, float, float],
    size: tuple[float, float, float],
    pivot: tuple[float, float, float],
    flag: str,
    small: bool,
) -> np.ndarray:
    scale, lift = (0.75, 16.0) if flag == "head" and small else (1.0, 0.0)
    if small and flag != "head":
        scale, lift = 0.5, 24.0
    px, py, pz = (
        pivot[0] * scale,
        (pivot[1] + lift) * scale,
        pivot[2] * scale,
    )
    lo = np.array(box, np.float32) * scale
    hi = lo + np.array(size, np.float32) * scale
    x0, x1 = 8.0 + px + lo[0], 8.0 + px + hi[0]
    z0, z1 = 8.0 + pz + lo[2], 8.0 + pz + hi[2]
    y0, y1 = 24.0 - (py + hi[1]), 24.0 - (py + lo[1])
    u, v = uv
    dz, dx, dy = size[2] * scale, size[0] * scale, size[1] * scale
    rects = {
        "up": (u + dz, v, u + dz + dx, v + dz),
        "down": (u + dz + dx, v, u + dz + 2 * dx, v + dz),
        "west": (u, v + dz, u + dz, v + dz + dy),
        "north": (u + dz, v + dz, u + dz + dx, v + dz + dy),
        "east": (u + dz + dx, v + dz, u + dz + dx + dz, v + dz + dy),
        "south": (u + dz + dx + dz, v + dz, u + dz + dx + dz + dx, v + dz + dy),
    }
    corners = {
        "down": ((x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0)),
        "up": ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)),
        "north": ((x1, y1, z0), (x0, y1, z0), (x0, y0, z0), (x1, y0, z0)),
        "south": ((x0, y1, z1), (x1, y1, z1), (x1, y0, z1), (x0, y0, z1)),
        "west": ((x0, y1, z1), (x0, y1, z0), (x0, y0, z0), (x0, y0, z1)),
        "east": ((x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (x1, y0, z0)),
    }
    quads = np.zeros(6, QUAD_DTYPE)
    for i, (face, tier) in enumerate(_PART_FACES):
        fu0, fv0, fu1, fv1 = (c / 4.0 for c in rects[face])
        quads["vtx"][i] = corners[face]
        quads["uv"][i] = ((fu0, fv0), (fu1, fv0), (fu1, fv1), (fu0, fv1))
        quads["texkey"][i] = _ARMORSTAND_TEXKEY
        quads["shade"][i] = tier
        quads["dbl"][i] = True
        quads["cull"][i] = CULL_NONE
    return quads
