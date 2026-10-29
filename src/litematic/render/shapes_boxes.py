"""Box primitive and single-box builtin families for the shape table."""

from collections.abc import Mapping

import numpy as np

from litematic.render.quads import (
    QUAD_DTYPE,
    TIER_DOWN,
    TIER_NORTH,
    TIER_SOUTH,
    TIER_UP,
    TIER_WEST,
)

__all__ = ["box_quads"]

Box = tuple[float, float, float, float, float, float]
_FACINGS = ("north", "south", "east", "west")

# Face corner templates per tier (down/up/north/south/west/east) in the
# pinned winding, mirroring quads._CUBE_VTX so a full-cube box matches the
# quad core exactly (kept in sync by test_shapes).
_FACE_CORNERS: tuple[tuple[tuple[int, int, int], ...], ...] = (
    ((0, 0, 16), (16, 0, 16), (16, 0, 0), (0, 0, 0)),  # down
    ((0, 16, 0), (16, 16, 0), (16, 16, 16), (0, 16, 16)),  # up
    ((16, 16, 0), (0, 16, 0), (0, 0, 0), (16, 0, 0)),  # north
    ((0, 16, 16), (16, 16, 16), (16, 0, 16), (0, 0, 16)),  # south
    ((0, 16, 16), (0, 16, 0), (0, 0, 0), (0, 0, 16)),  # west
    ((16, 16, 0), (16, 16, 16), (16, 0, 16), (16, 0, 0)),  # east
)

# Panels keyed by facing: open trapdoors rest on the facing edge, closed
# doors stand in the facing plane, open doors swing to the side, ladders
# and wall torches hug the facing wall (Blockscope shape.rs taxonomy).
_OPEN_TRAPDOOR: dict[str, Box] = {
    "north": (0, 0, 0, 16, 16, 3),
    "south": (0, 0, 13, 16, 16, 16),
    "west": (0, 0, 0, 3, 16, 16),
    "east": (13, 0, 0, 16, 16, 16),
}
_DOOR_CLOSED: dict[str, Box] = {
    "north": (0, 0, 0, 16, 16, 3),
    "south": (0, 0, 13, 16, 16, 16),
    "west": (0, 0, 0, 3, 16, 16),
    "east": (13, 0, 0, 16, 16, 16),
}
_DOOR_OPEN: dict[str, Box] = {
    "north": (0, 0, 0, 3, 16, 16),
    "south": (13, 0, 0, 16, 16, 16),
    "west": (0, 0, 13, 16, 16, 16),
    "east": (0, 0, 0, 16, 16, 3),
}
_LADDER: dict[str, Box] = {
    "north": (0, 0, 0, 16, 16, 2),
    "south": (0, 0, 14, 16, 16, 16),
    "west": (0, 0, 0, 2, 16, 16),
    "east": (14, 0, 0, 16, 16, 16),
}
_BUTTON_WALL: dict[str, Box] = {
    "north": (5, 5, 0, 11, 11, 2),
    "south": (5, 5, 14, 11, 11, 16),
    "west": (0, 5, 5, 2, 11, 11),
    "east": (14, 5, 5, 16, 11, 11),
}
_TORCH_WALL: dict[str, Box] = {
    "north": (7, 3, 0, 9, 13, 2),
    "south": (7, 3, 14, 9, 13, 16),
    "west": (0, 3, 7, 2, 13, 9),
    "east": (14, 3, 7, 16, 13, 9),
}
# Stair step footprint (x0, x1, z0, z1) per facing; north is the default.
_STAIR_STEP: dict[str, tuple[int, int, int, int]] = {
    "north": (0, 16, 0, 8),
    "south": (0, 16, 8, 16),
    "west": (0, 8, 0, 16),
    "east": (8, 16, 0, 16),
}


def box_quads(
    x0: float, y0: float, z0: float, x1: float, y1: float, z1: float
) -> np.ndarray:
    """Build the six textured faces of one axis-aligned box with auto-UV."""
    lo = (float(x0), float(y0), float(z0))
    hi = (float(x1), float(y1), float(z1))
    quads = np.empty(6, dtype=QUAD_DTYPE)
    for tier in range(6):
        points = [
            tuple(lo[axis] if corner[axis] == 0 else hi[axis] for axis in range(3))
            for corner in _FACE_CORNERS[tier]
        ]
        quads["vtx"][tier] = points
        quads["uv"][tier] = [_auto_uv(tier, *point) for point in points]
        quads["shade"][tier] = tier
    quads["texkey"] = ""
    quads["dbl"] = False
    return quads


def _auto_uv(tier: int, x: float, y: float, z: float) -> tuple[float, float]:
    """Map one box corner to texel coords per the pinned auto-UV axes."""
    if tier == TIER_DOWN:
        return (x, 16.0 - z)
    if tier == TIER_UP:
        return (x, z)
    if tier == TIER_NORTH:
        return (16.0 - x, 16.0 - y)
    if tier == TIER_SOUTH:
        return (x, 16.0 - y)
    if tier == TIER_WEST:
        return (16.0 - z, 16.0 - y)
    return (z, 16.0 - y)


def _combine(boxes: list[Box]) -> np.ndarray:
    """Concatenate the six-face quads of several boxes into one array."""
    return np.concatenate([box_quads(*box) for box in boxes])


def _text(properties: Mapping[str, str], key: str, default: str) -> str:
    """Read one property as a string, falling back when missing or odd."""
    value = properties.get(key, default)
    return value if isinstance(value, str) else default


def _flag(properties: Mapping[str, str], key: str) -> bool:
    """Read one property as a boolean flag (only "true" counts)."""
    return _text(properties, key, "false") == "true"


def _facing(properties: Mapping[str, str]) -> str:
    """Read the facing property, limited to the four horizontal names."""
    value = _text(properties, "facing", "north")
    return value if value in _FACINGS else "north"


def _count(
    properties: Mapping[str, str], key: str, default: int, lo: int, hi: int
) -> int:
    """Read one property as an integer clamped into [lo, hi]."""
    raw = _text(properties, key, "")
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(lo, min(hi, value))


def _fluid(properties: Mapping[str, str]) -> np.ndarray:
    """Water, lava and bubble columns: level drops the surface."""
    level = _count(properties, "level", 0, 0, 15)
    return box_quads(0, 0, 0, 16, max(1, 16 - level), 16)


def _snow(properties: Mapping[str, str]) -> np.ndarray:
    """Snow layers: each layer adds two texels of height."""
    layers = _count(properties, "layers", 1, 1, 8)
    return box_quads(0, 0, 0, 16, layers * 2, 16)


def _slab(properties: Mapping[str, str]) -> np.ndarray:
    """Slab: bottom/top half box; double is a full cube."""
    kind = _text(properties, "type", "bottom")
    if kind == "double":
        return box_quads(0, 0, 0, 16, 16, 16)
    if kind == "top":
        return box_quads(0, 8, 0, 16, 16, 16)
    return box_quads(0, 0, 0, 16, 8, 16)


def _stairs(properties: Mapping[str, str]) -> np.ndarray:
    """Stairs: full-width base half slab plus a facing step box."""
    top = _text(properties, "half", "bottom") == "top"
    base: Box = (0, 8, 0, 16, 16, 16) if top else (0, 0, 0, 16, 8, 16)
    step_y = (0.0, 8.0) if top else (8.0, 16.0)
    x0, x1, z0, z1 = _STAIR_STEP[_facing(properties)]
    return _combine([base, (x0, step_y[0], z0, x1, step_y[1], z1)])


def _carpet(properties: Mapping[str, str]) -> np.ndarray:
    """Carpet: one-texel-thick full-footprint sheet."""
    return box_quads(0, 0, 0, 16, 1, 16)


def _rail(properties: Mapping[str, str]) -> np.ndarray:
    """Rail: one-texel-thick full-footprint sheet."""
    return box_quads(0, 0, 0, 16, 1, 16)


def _pressure_plate(properties: Mapping[str, str]) -> np.ndarray:
    """Pressure plate: one-texel-thick inset sheet."""
    return box_quads(1, 0, 1, 15, 1, 15)


def _repeater(properties: Mapping[str, str]) -> np.ndarray:
    """Repeater and comparator: low inset panel."""
    return box_quads(2, 0, 2, 14, 2, 14)


def _trapdoor(properties: Mapping[str, str]) -> np.ndarray:
    """Trapdoor: thin slab at the half plane, edge panel when open."""
    if _flag(properties, "open"):
        return box_quads(*_OPEN_TRAPDOOR[_facing(properties)])
    if _text(properties, "half", "bottom") == "top":
        return box_quads(0, 13, 0, 16, 16, 16)
    return box_quads(0, 0, 0, 16, 3, 16)


def _door(properties: Mapping[str, str]) -> np.ndarray:
    """Door: three-texel facing panel, swung sideways when open."""
    table = _DOOR_OPEN if _flag(properties, "open") else _DOOR_CLOSED
    return box_quads(*table[_facing(properties)])


def _ladder(properties: Mapping[str, str]) -> np.ndarray:
    """Ladder: two-texel panel flush against the facing wall."""
    return box_quads(*_LADDER[_facing(properties)])


def _button(properties: Mapping[str, str]) -> np.ndarray:
    """Button: small pad on the floor, ceiling, or facing wall."""
    face = _text(properties, "face", "floor")
    if face == "ceiling":
        return box_quads(5, 14, 5, 11, 16, 11)
    if face == "wall":
        return box_quads(*_BUTTON_WALL[_facing(properties)])
    return box_quads(5, 0, 5, 11, 2, 11)


def _sign(properties: Mapping[str, str]) -> np.ndarray:
    """Sign (standing or wall): two-texel post slab."""
    return box_quads(7, 0, 0, 9, 16, 16)


def _banner(properties: Mapping[str, str]) -> np.ndarray:
    """Banner (standing or wall): two-texel post slab."""
    return box_quads(7, 0, 0, 9, 16, 16)


def _chest(properties: Mapping[str, str]) -> np.ndarray:
    """Chest family: 14-texel box leaving seam margins."""
    return box_quads(1, 0, 1, 15, 14, 15)


def _shulker_box(properties: Mapping[str, str]) -> np.ndarray:
    """Shulker box: slightly inset full-height box."""
    return box_quads(1, 0, 1, 15, 16, 15)


def _barrel(properties: Mapping[str, str]) -> np.ndarray:
    """Barrel: slightly inset full-height box."""
    return box_quads(1, 0, 1, 15, 16, 15)


def _composter(properties: Mapping[str, str]) -> np.ndarray:
    """Composter: slightly inset full-height box."""
    return box_quads(1, 0, 1, 15, 16, 15)


def _head(properties: Mapping[str, str]) -> np.ndarray:
    """Head and skull: centered 8x8x8 box."""
    return box_quads(4, 0, 4, 12, 8, 12)


def _cactus(properties: Mapping[str, str]) -> np.ndarray:
    """Cactus: inset full-height column."""
    return box_quads(1, 0, 1, 15, 16, 15)


def _lily_pad(properties: Mapping[str, str]) -> np.ndarray:
    """Lily pad: one-texel-thick full-footprint sheet."""
    return box_quads(0, 0, 0, 16, 1, 16)


def _conduit(properties: Mapping[str, str]) -> np.ndarray:
    """Conduit: centered 8x8x8 core."""
    return box_quads(4, 4, 4, 12, 12, 12)


def _lectern(properties: Mapping[str, str]) -> np.ndarray:
    """Lectern: full-footprint base plus a centered desk box."""
    return _combine([(0, 0, 0, 16, 4, 16), (4, 4, 4, 12, 12, 12)])


def _torch_standing(properties: Mapping[str, str]) -> np.ndarray:
    """Standing torch: small centered post."""
    return box_quads(7, 0, 7, 9, 10, 9)


def _torch_wall(properties: Mapping[str, str]) -> np.ndarray:
    """Wall torch: small post lying against the facing wall."""
    return box_quads(*_TORCH_WALL[_facing(properties)])
