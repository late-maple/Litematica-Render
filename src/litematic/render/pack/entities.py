"""Block-entity-shaped blocks: chests compiled from entity textures."""

import numpy as np

from litematic.render.pack.blockstate import _state_pairs
from litematic.render.quads import (
    CULL_NONE,
    QUAD_DTYPE,
    TIER_DOWN,
    TIER_EAST,
    TIER_NORTH,
    TIER_SOUTH,
    TIER_UP,
    TIER_WEST,
)

_CHEST_BASES = {
    "chest": "normal",
    "trapped_chest": "trapped",
    "ender_chest": "ender",
    "copper_chest": "copper",
    "waxed_copper_chest": "copper",
    "exposed_copper_chest": "copper_exposed",
    "waxed_exposed_copper_chest": "copper_exposed",
    "weathered_copper_chest": "copper_weathered",
    "waxed_weathered_copper_chest": "copper_weathered",
    "oxidized_copper_chest": "copper_oxidized",
    "waxed_oxidized_copper_chest": "copper_oxidized",
}
# The model's front is +Z (south); the rotation turns it to the facing.
_FACING_ROTATION = {"south": 0, "east": 90, "north": 180, "west": 270}
_COS_SIN = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}

_TIER_OF_NORMAL = {
    (0, 1, 0): TIER_UP,
    (0, -1, 0): TIER_DOWN,
    (0, 0, -1): TIER_NORTH,
    (0, 0, 1): TIER_SOUTH,
    (-1, 0, 0): TIER_WEST,
    (1, 0, 0): TIER_EAST,
}


def chest_quads(
    block_id: str,
    state_key: str,
    resolve_texture,
) -> np.ndarray | None:
    """Compile one closed chest from entity textures, or None if uncovered.

    `resolve_texture` maps an entity texture ref (entity/chest/...) to a
    registered texkey, returning None when the pack lacks the file.
    """
    stem = block_id.partition(":")[2]
    base = _CHEST_BASES.get(stem)
    if base is None:
        return None
    properties = _state_pairs(state_key)
    kind = properties.get("type", "single")
    if kind not in ("single", "left", "right"):
        return None
    texkey = resolve_texture(_chest_texture_ref(base, kind))
    if texkey is None:
        return None
    degrees = _FACING_ROTATION.get(properties.get("facing", "north"), 180)

    faces: list[tuple[list, list, tuple[int, int, int], str]] = []
    if kind == "single":
        _add_box(faces, (1, 0, 1), (15, 10, 15), (0, 19), texkey)
        _add_box(faces, (1, 9, 1), (15, 14, 15), (0, 0), texkey)
        _add_box(faces, (7, 7, 15), (9, 11, 16), (0, 0), texkey)
    else:
        x1, x2 = (0, 15) if kind == "left" else (1, 16)
        _add_box(faces, (x1, 0, 1), (x2, 10, 15), (0, 19), texkey)
        _add_box(faces, (x1, 9, 1), (x2, 14, 15), (0, 0), texkey)
        lock_x = 0 if kind == "left" else 15
        _add_box(faces, (lock_x, 7, 15), (lock_x + 1, 11, 16), (0, 0), texkey)

    quads = np.empty(len(faces), dtype=QUAD_DTYPE)
    for index, (vertices, uvs, normal, face_texkey) in enumerate(faces):
        rotated = [_rotate_y(point, degrees) for point in vertices]
        quads[index] = (
            rotated,
            [(u / 4.0, v / 4.0) for (u, v) in uvs],
            face_texkey,
            _TIER_OF_NORMAL[_rotate_normal(normal, degrees)],
            False,
            CULL_NONE,
        )
    return quads


def _chest_texture_ref(base: str, kind: str) -> str:
    if kind in ("left", "right"):
        return f"entity/chest/{base}_{kind}"
    return f"entity/chest/{base}"


def _add_box(
    faces: list,
    frm: tuple[int, int, int],
    to: tuple[int, int, int],
    origin: tuple[int, int],
    texkey: str,
) -> None:
    """Append one box's six faces in entity-unwrap order and UV layout."""
    x1, y1, z1 = frm
    x2, y2, z2 = to
    width, height, depth = x2 - x1, y2 - y1, z2 - z1
    u, v = origin
    x1f, y1f, z1f, x2f, y2f, z2f = (
        float(x1), float(y1), float(z1), float(x2), float(y2), float(z2)
    )
    uf, vf, wf, hf, df = (
        float(u), float(v), float(width), float(height), float(depth)
    )
    faces.extend([
        (
            [(x1f, y1f, z2f), (x1f, y1f, z1f), (x2f, y1f, z1f), (x2f, y1f, z2f)],
            [(uf + df, vf), (uf + df, vf + df),
             (uf + df + wf, vf + df), (uf + df + wf, vf)],
            (0, -1, 0),
            texkey,
        ),
        (
            [(x1f, y2f, z1f), (x1f, y2f, z2f), (x2f, y2f, z2f), (x2f, y2f, z1f)],
            [(uf + df + wf, vf + df), (uf + df + wf, vf),
             (uf + df + 2 * wf, vf), (uf + df + 2 * wf, vf + df)],
            (0, 1, 0),
            texkey,
        ),
        (
            [(x1f, y1f, z1f), (x1f, y2f, z1f), (x2f, y2f, z1f), (x2f, y1f, z1f)],
            [(uf + df, vf + df), (uf + df, vf + df + hf),
             (uf + df + wf, vf + df + hf), (uf + df + wf, vf + df)],
            (0, 0, -1),
            texkey,
        ),
        (
            [(x2f, y1f, z2f), (x2f, y2f, z2f), (x1f, y2f, z2f), (x1f, y1f, z2f)],
            [(uf + df + wf + df, vf + df), (uf + df + wf + df, vf + df + hf),
             (uf + df + 2 * wf + df, vf + df + hf),
             (uf + df + 2 * wf + df, vf + df)],
            (0, 0, 1),
            texkey,
        ),
        (
            [(x1f, y1f, z2f), (x1f, y2f, z2f), (x1f, y2f, z1f), (x1f, y1f, z1f)],
            [(uf, vf + df), (uf, vf + df + hf),
             (uf + df, vf + df + hf), (uf + df, vf + df)],
            (-1, 0, 0),
            texkey,
        ),
        (
            [(x2f, y1f, z1f), (x2f, y2f, z1f), (x2f, y2f, z2f), (x2f, y1f, z2f)],
            [(uf + df + wf, vf + df), (uf + df + wf, vf + df + hf),
             (uf + df + wf + df, vf + df + hf), (uf + df + wf + df, vf + df)],
            (1, 0, 0),
            texkey,
        ),
    ])


def _rotate_y(
    point: tuple[float, float, float], degrees: int
) -> tuple[float, float, float]:
    """Rotate one block-space point about the y axis through (8, *, 8)."""
    cos, sin = _COS_SIN[degrees]
    x, y, z = point[0] - 8.0, point[1], point[2] - 8.0
    return (x * cos + z * sin + 8.0, y, -x * sin + z * cos + 8.0)


def _rotate_normal(
    normal: tuple[int, int, int], degrees: int
) -> tuple[int, int, int]:
    cos, sin = _COS_SIN[degrees]
    return (
        round(normal[0] * cos + normal[2] * sin),
        normal[1],
        round(-normal[0] * sin + normal[2] * cos),
    )
