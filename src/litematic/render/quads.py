"""Quad data core: dtype, face tiers, unit cubes, ordering, geometry checks."""

import numpy as np

__all__ = [
    "AO_LEVELS",
    "CULL_NONE",
    "FACE_LIGHT",
    "QUAD_DTYPE",
    "TIER_DOWN",
    "TIER_EAST",
    "TIER_NORTH",
    "TIER_SOUTH",
    "TIER_UP",
    "TIER_WEST",
    "is_parallelogram",
    "sort_quads",
    "unit_cube_quads",
]

QUAD_DTYPE = np.dtype(
    [
        ("vtx", "f4", (4, 3)),
        ("uv", "f4", (4, 2)),
        ("texkey", "U64"),
        ("shade", "u1"),
        ("dbl", "?"),
        ("cull", "u1"),
    ]
)

# No cullface direction: the quad never disappears through neighbor culling.
CULL_NONE = 255

TIER_DOWN = 0
TIER_UP = 1
TIER_NORTH = 2
TIER_SOUTH = 3
TIER_WEST = 4
TIER_EAST = 5

FACE_LIGHT = (0.5, 1.0, 0.8, 0.8, 0.6, 0.6)
AO_LEVELS = (0.4, 0.6, 0.8, 1.0)

# Per-tier corners in the pinned winding; auto-UVs follow vanilla defaultFaceUV.
_CUBE_VTX = np.array(
    [
        [(0, 0, 16), (16, 0, 16), (16, 0, 0), (0, 0, 0)],  # down
        [(0, 16, 0), (16, 16, 0), (16, 16, 16), (0, 16, 16)],  # up
        [(16, 16, 0), (0, 16, 0), (0, 0, 0), (16, 0, 0)],  # north
        [(0, 16, 16), (16, 16, 16), (16, 0, 16), (0, 0, 16)],  # south
        [(0, 16, 16), (0, 16, 0), (0, 0, 0), (0, 0, 16)],  # west
        [(16, 16, 0), (16, 16, 16), (16, 0, 16), (16, 0, 0)],  # east
    ],
    dtype="f4",
)
_CUBE_UV = np.tile(
    np.array([(0, 0), (16, 0), (16, 16), (0, 16)], dtype="f4"), (6, 1, 1)
)
# West/east corners run the opposite winding, hence the flipped uv ring.
_CUBE_UV[TIER_WEST] = [(16, 0), (0, 0), (0, 16), (16, 16)]
_CUBE_UV[TIER_EAST] = [(16, 0), (0, 0), (0, 16), (16, 16)]


def unit_cube_quads(texkey: str, shade_faces: bool = True) -> np.ndarray:
    """Build the six textured faces of a full 0..16 unit cube."""
    quads = np.empty(6, dtype=QUAD_DTYPE)
    quads["vtx"] = _CUBE_VTX
    quads["uv"] = _CUBE_UV
    quads["texkey"] = texkey
    if shade_faces:
        quads["shade"] = np.arange(6, dtype=np.uint8)
    else:
        quads["shade"] = np.full(6, TIER_UP, dtype=np.uint8)
    quads["dbl"] = False
    quads["cull"] = CULL_NONE
    return quads


def sort_quads(quads: np.ndarray) -> np.ndarray:
    """Return a canonical copy ordered by the raw byte sequence of vtx."""
    vtx = np.ascontiguousarray(quads["vtx"]).reshape(quads.shape[0], 12)
    keys = vtx.view(np.dtype((np.void, vtx.dtype.itemsize * 12))).reshape(-1)
    order = np.argsort(keys, kind="stable")
    return quads[order]


def is_parallelogram(quads: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Flag quads whose diagonal midpoints coincide within eps on each axis."""
    vtx = quads["vtx"]
    diag = (vtx[:, 0] + vtx[:, 2]) - (vtx[:, 1] + vtx[:, 3])
    return np.all(np.abs(diag) <= eps, axis=-1)

_REDSTONE_LOW = (76, 0, 0)
_REDSTONE_HIGH = (255, 26, 26)


def _redstone_tint(power: int) -> tuple[int, int, int]:
    """Vanilla dust color for a redstone power level, 0-15."""
    factor = max(0, min(15, power)) / 15.0
    low, high = _REDSTONE_LOW, _REDSTONE_HIGH
    return (
        round(low[0] + (high[0] - low[0]) * factor),
        round(low[1] + (high[1] - low[1]) * factor),
        round(low[2] + (high[2] - low[2]) * factor),
    )
