"""Builtin shape table: box families plus multi-box and cross plants."""

from collections.abc import Callable, Mapping

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.quads import (
    CULL_NONE,
    QUAD_DTYPE,
    TIER_EAST,
    TIER_NORTH,
    TIER_SOUTH,
    TIER_UP,
    TIER_WEST,
    _redstone_tint,
)
from litematic.render.shapes.boxes import (
    Box,
    _banner,
    _barrel,
    _button,
    _cactus,
    _carpet,
    _chest,
    _combine,
    _composter,
    _conduit,
    _door,
    _facing,
    _flag,
    _fluid,
    _head,
    _ladder,
    _lectern,
    _lily_pad,
    _pressure_plate,
    _rail,
    _repeater,
    _shulker_box,
    _sign,
    _slab,
    _snow,
    _stairs,
    _text,
    _torch_standing,
    _torch_wall,
    _trapdoor,
    box_quads,
)

__all__ = ["builtin_quads", "box_quads"]

Builder = Callable[[Mapping[str, str]], np.ndarray]
_SIDES = ("north", "south", "west", "east")

# Connection geometry per family (Blockscope shape.rs taxonomy): fences
# carry two rails per connected side at y 6..9 and 12..15, walls one full
# arm with the post height driven by "up" (low and tall arms share the
# same 0..14 box, so tall arms' cross-block reach is approximated), panes
# and iron bars one full arm per side plus a full bar when unconnected.
_FENCE_POST: Box = (6, 0, 6, 10, 16, 10)
_FENCE_ARMS: dict[str, tuple[Box, Box]] = {
    "north": ((6, 6, 0, 10, 9, 6), (6, 12, 0, 10, 15, 6)),
    "south": ((6, 6, 10, 10, 9, 16), (6, 12, 10, 10, 15, 16)),
    "west": ((0, 6, 6, 6, 9, 10), (0, 12, 6, 6, 15, 10)),
    "east": ((10, 6, 6, 16, 9, 10), (10, 12, 6, 16, 15, 10)),
}
_GATE_POST: Box = (6, 0, 6, 10, 16, 10)
# Open gate bars stay centered over the full block depth per Blockscope,
# coplanarly overlapping the post side faces; both faces share the same
# auto-UV texel mapping, so the overlap does not change rendered pixels.
_GATE_BAR_X: Box = (0, 6, 6, 16, 15, 10)
_GATE_BAR_Z: Box = (6, 6, 0, 10, 15, 16)
_WALL_ARMS: dict[str, Box] = {
    "north": (4, 0, 0, 12, 14, 4),
    "south": (4, 0, 12, 12, 14, 16),
    "west": (0, 0, 4, 4, 14, 12),
    "east": (12, 0, 4, 16, 14, 12),
}
_PANE_POST: Box = (7, 0, 7, 9, 16, 9)
_PANE_ARMS: dict[str, Box] = {
    "north": (7, 0, 0, 9, 16, 7),
    "south": (7, 0, 9, 9, 16, 16),
    "west": (0, 0, 7, 7, 16, 9),
    "east": (9, 0, 7, 16, 16, 9),
}
_PANE_BAR: Box = (7, 0, 0, 9, 16, 16)

# Cross plants: two diagonal planes spanning the full block height,
# texture v running bottom (v=16) to top (v=0); both quads are double-
# sided and carry the up tier like the vanilla cross model.
_CROSS_VTX = np.array(
    [
        [(0, 0, 0), (16, 0, 16), (16, 16, 16), (0, 16, 0)],
        [(16, 0, 0), (0, 0, 16), (0, 16, 16), (16, 16, 0)],
    ],
    dtype="f4",
)
_CROSS_UV = np.tile(
    np.array([(0, 16), (16, 16), (16, 0), (0, 0)], dtype="f4"), (2, 1, 1)
)

# Cross-plant block names rendered as two diagonal planes (Blockscope
# taxonomy plus the sapling/propagule suffix families).
_CROSS_PLANTS = frozenset(
    {
        "short_grass",
        "tall_grass",
        "fern",
        "large_fern",
        "dandelion",
        "poppy",
        "blue_orchid",
        "allium",
        "azure_bluet",
        "oxeye_daisy",
        "cornflower",
        "lily_of_the_valley",
        "red_tulip",
        "orange_tulip",
        "white_tulip",
        "pink_tulip",
        "sunflower",
        "lilac",
        "rose_bush",
        "peony",
        "vine",
        "cobweb",
        "bamboo",
        "dead_bush",
        "seagrass",
        "kelp",
        "kelp_plant",
        "sugar_cane",
        "wheat",
        "carrots",
        "potatoes",
        "beetroots",
        "torchflower",
        "pitcher_plant",
        "bush",
        "sweet_berry_bush",
        "tall_seagrass",
        "red_mushroom",
        "brown_mushroom",
        "crimson_roots",
        "warped_roots",
        "nether_sprouts",
        "crimson_fungus",
        "warped_fungus",
    }
)


def _fence(properties: Mapping[str, str]) -> np.ndarray:
    """Fence: center post plus two rails on every connected side."""
    boxes: list[Box] = [_FENCE_POST]
    for side in _SIDES:
        if _flag(properties, side):
            boxes.extend(_FENCE_ARMS[side])
    return _combine(boxes)


def _fence_gate(properties: Mapping[str, str]) -> np.ndarray:
    """Fence gate: center post plus a bar swung aside when open."""
    along_x = _facing(properties) in ("north", "south")
    bar = _GATE_BAR_X if along_x else _GATE_BAR_Z
    if _flag(properties, "open"):
        bar = _GATE_BAR_Z if along_x else _GATE_BAR_X
    return _combine([_GATE_POST, bar])


def _side(properties: Mapping[str, str], key: str) -> bool:
    """True when a wall side property is present and not "none"."""
    return _text(properties, key, "none") != "none"


def _wall(properties: Mapping[str, str]) -> np.ndarray:
    """Wall: center column plus one arm per connected side."""
    top = 16.0 if _flag(properties, "up") else 14.0
    boxes: list[Box] = [(4, 0, 4, 12, top, 12)]
    for side in _SIDES:
        if _side(properties, side):
            boxes.append(_WALL_ARMS[side])
    return _combine(boxes)


def _pane(properties: Mapping[str, str]) -> np.ndarray:
    """Pane and iron bars: center post plus arms, full bar when alone."""
    boxes: list[Box] = [_PANE_POST]
    connected = False
    for side in _SIDES:
        if _flag(properties, side):
            boxes.append(_PANE_ARMS[side])
            connected = True
    if not connected:
        boxes.append(_PANE_BAR)
    return _combine(boxes)


def _cross(properties: Mapping[str, str]) -> np.ndarray:
    """Cross plant: two diagonal double-sided planes at full height."""
    quads = np.empty(2, dtype=QUAD_DTYPE)
    quads["vtx"] = _CROSS_VTX
    quads["uv"] = _CROSS_UV
    quads["texkey"] = ""
    quads["shade"] = TIER_UP
    quads["dbl"] = True
    quads["cull"] = CULL_NONE
    return quads


def _named(*names: str) -> Callable[[str], bool]:
    """Match one of the exact block names."""
    return lambda stem: stem in names


def _suffixed(suffix: str) -> Callable[[str], bool]:
    """Match any block name ending with the suffix."""
    return lambda stem: stem.endswith(suffix)


def _is_shulker(stem: str) -> bool:
    """Match the shulker box family including the base block."""
    return stem == "shulker_box" or stem.endswith("_shulker_box")


def _is_head(stem: str) -> bool:
    """Match heads and skulls."""
    return stem.endswith("_head") or stem.endswith("skull")


def _is_wall_torch(stem: str) -> bool:
    """Match wall-mounted torches."""
    return stem == "wall_torch" or stem.endswith("_wall_torch")


def _is_torch(stem: str) -> bool:
    """Match standing torches (checked after wall torches)."""
    return stem == "torch" or stem.endswith("_torch")


def _is_pane(stem: str) -> bool:
    """Match glass panes and the iron bars block."""
    return stem == "iron_bars" or stem.endswith("_pane")


def _is_cross(stem: str) -> bool:
    """Match cross-plant names and sapling/propagule suffixes."""
    return stem in _CROSS_PLANTS or stem.endswith(("_sapling", "_propagule"))


def _ground_plane(transposed: bool, texkey: str) -> np.ndarray:
    """One full-width dust plane on the block floor; transposed uv runs E-W."""
    quad = box_quads(0, 0, 0, 16, 0, 16)[[TIER_UP]]
    if transposed:
        quad["uv"][0] = quad["vtx"][0][:, [2, 0]]
    quad["texkey"][0] = texkey
    return quad


def _wall_strip(tier: int, texkey: str) -> np.ndarray:
    """One full-height dust strip on a block face, visible from both sides."""
    bounds = {
        TIER_NORTH: (0, 0, 0, 16, 16, 0),
        TIER_SOUTH: (0, 0, 16, 16, 16, 16),
        TIER_WEST: (0, 0, 0, 0, 16, 16),
        TIER_EAST: (16, 0, 0, 16, 16, 16),
    }[tier]
    quad = box_quads(*bounds)[[tier]]
    quad["texkey"][0] = texkey
    quad["dbl"][0] = True
    quad["shade"][0] = TIER_UP
    return quad


def _wire(properties: Mapping[str, str]) -> np.ndarray:
    """Redstone dust: ground planes per connected axis, wall strips for up."""
    try:
        power = max(0, min(15, int(properties.get("power", "0"))))
    except (TypeError, ValueError):
        power = 0
    r, g, b = _redstone_tint(power)
    line = f"block/redstone_dust_line0@{r},{g},{b}"
    dot = f"block/redstone_dust_dot@{r},{g},{b}"
    connected = [
        properties.get(direction, "none") in ("side", "up")
        for direction in _SIDES
    ]
    quads: list[np.ndarray] = []
    if not any(connected):
        quads.append(_ground_plane(False, dot))
    else:
        # North-south planes sample the texture column along x; east-west
        # planes transpose the uv so the same column runs along z.
        if connected[0] or connected[1]:
            quads.append(_ground_plane(False, line))
        if connected[2] or connected[3]:
            quads.append(_ground_plane(True, line))
    for direction, tier in zip(_SIDES, (TIER_NORTH, TIER_SOUTH, TIER_WEST, TIER_EAST)):
        if properties.get(direction, "none") == "up":
            quads.append(_wall_strip(tier, line))
    return np.concatenate(quads)


# Ordered dispatch, earlier rows win: wall torches must precede standing
# torches because "*_wall_torch" also ends with "_torch". The other rows
# are pairwise disjoint and keep Blockscope's precedence for reference.
_FAMILIES: tuple[tuple[Callable[[str], bool], Builder], ...] = (
    (_suffixed("_fence_gate"), _fence_gate),
    (_suffixed("_fence"), _fence),
    (_suffixed("_wall"), _wall),
    (_is_pane, _pane),
    (_is_cross, _cross),
    (_named("water", "lava", "bubble_column"), _fluid),
    (_named("snow"), _snow),
    (_named("redstone_wire"), _wire),
    (_named("composter"), _composter),
    (_suffixed("_slab"), _slab),
    (_suffixed("_stairs"), _stairs),
    (_suffixed("_carpet"), _carpet),
    (_suffixed("rail"), _rail),
    (_suffixed("_pressure_plate"), _pressure_plate),
    (_named("repeater", "comparator"), _repeater),
    (_suffixed("_trapdoor"), _trapdoor),
    (_suffixed("_door"), _door),
    (_named("ladder"), _ladder),
    (_suffixed("_button"), _button),
    (_suffixed("sign"), _sign),
    (_suffixed("banner"), _banner),
    (_named("chest", "trapped_chest", "ender_chest"), _chest),
    (_is_shulker, _shulker_box),
    (_named("barrel"), _barrel),
    (_is_head, _head),
    (_named("cactus"), _cactus),
    (_named("lily_pad"), _lily_pad),
    (_named("conduit"), _conduit),
    (_named("lectern"), _lectern),
    (_is_wall_torch, _torch_wall),
    (_is_torch, _torch_standing),
)


def builtin_quads(
    state_name: str, properties: Mapping[str, str]
) -> np.ndarray | None:
    """Build geometry-only quads for a known family, else None."""
    if not isinstance(state_name, str):
        raise RenderError("shapes: block state name must be a string")
    if not isinstance(properties, Mapping):
        raise RenderError("shapes: block properties must be a mapping")
    stem = state_name.partition(":")[2] or state_name
    for matches, build in _FAMILIES:
        if matches(stem):
            return build(properties)
    return None
