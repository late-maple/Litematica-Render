"""Quad data core contract: dtype, tiers, unit cube, ordering, geometry."""

import numpy as np

from litematic.render.quads import (
    AO_LEVELS,
    CULL_NONE,
    FACE_LIGHT,
    QUAD_DTYPE,
    TIER_DOWN,
    TIER_EAST,
    TIER_NORTH,
    TIER_SOUTH,
    TIER_UP,
    TIER_WEST,
    is_parallelogram,
    sort_quads,
    unit_cube_quads,
)

# Per-vertex auto-UV expectations per the vanilla defaultFaceUV axes.
_CUBE = {
    TIER_DOWN: (
        [(0, 0, 16), (16, 0, 16), (16, 0, 0), (0, 0, 0)],
        [(0, 0), (16, 0), (16, 16), (0, 16)],
    ),
    TIER_UP: (
        [(0, 16, 0), (16, 16, 0), (16, 16, 16), (0, 16, 16)],
        [(0, 0), (16, 0), (16, 16), (0, 16)],
    ),
    TIER_NORTH: (
        [(16, 16, 0), (0, 16, 0), (0, 0, 0), (16, 0, 0)],
        [(0, 0), (16, 0), (16, 16), (0, 16)],
    ),
    TIER_SOUTH: (
        [(0, 16, 16), (16, 16, 16), (16, 0, 16), (0, 0, 16)],
        [(0, 0), (16, 0), (16, 16), (0, 16)],
    ),
    TIER_WEST: (
        [(0, 16, 16), (0, 16, 0), (0, 0, 0), (0, 0, 16)],
        [(16, 0), (0, 0), (0, 16), (16, 16)],
    ),
    TIER_EAST: (
        [(16, 16, 0), (16, 16, 16), (16, 0, 16), (16, 0, 0)],
        [(16, 0), (0, 0), (0, 16), (16, 16)],
    ),
}


def _quad(vtx, uv=((0, 0), (16, 0), (16, 16), (0, 16))):
    quads = np.empty(1, dtype=QUAD_DTYPE)
    quads["vtx"][0] = vtx
    quads["uv"][0] = uv
    quads["texkey"][0] = "t"
    quads["shade"][0] = TIER_UP
    quads["dbl"][0] = False
    return quads


def test_tier_constants_match_vanilla_tables():
    tiers = (TIER_DOWN, TIER_UP, TIER_NORTH, TIER_SOUTH, TIER_WEST, TIER_EAST)
    assert tiers == (0, 1, 2, 3, 4, 5)
    assert FACE_LIGHT == (0.5, 1.0, 0.8, 0.8, 0.6, 0.6)
    assert AO_LEVELS == (0.4, 0.6, 0.8, 1.0)


def test_quad_dtype_fields():
    assert QUAD_DTYPE.names == ("vtx", "uv", "texkey", "shade", "dbl", "cull")
    assert QUAD_DTYPE["vtx"].shape == (4, 3)
    assert QUAD_DTYPE["vtx"].base == np.dtype("f4")
    assert QUAD_DTYPE["uv"].shape == (4, 2)
    assert QUAD_DTYPE["uv"].base == np.dtype("f4")
    assert QUAD_DTYPE["texkey"] == np.dtype("U64")
    assert QUAD_DTYPE["shade"] == np.dtype("u1")
    assert QUAD_DTYPE["dbl"] == np.dtype("?")
    assert QUAD_DTYPE["cull"] == np.dtype("u1")
    assert QUAD_DTYPE.itemsize == 4 * 12 + 4 * 8 + 256 + 1 + 1 + 1


def test_unit_cube_quads_covers_six_tiers():
    quads = unit_cube_quads("block/stone")
    assert quads.dtype == QUAD_DTYPE
    assert len(quads) == 6
    assert sorted(quads["shade"].tolist()) == [0, 1, 2, 3, 4, 5]
    assert quads["vtx"].min() == 0.0
    assert quads["vtx"].max() == 16.0
    assert quads["uv"].min() == 0.0
    assert quads["uv"].max() == 16.0
    assert not quads["dbl"].any()
    assert (quads["texkey"] == "block/stone").all()


def test_unit_cube_quads_match_auto_uv_table():
    quads = unit_cube_quads("block/stone")
    for tier, (vtx, uv) in _CUBE.items():
        row = quads[quads["shade"] == tier][0]
        assert np.array_equal(row["vtx"], np.array(vtx, dtype="f4"))
        assert np.array_equal(row["uv"], np.array(uv, dtype="f4"))


def test_unit_cube_winding_handedness_pinned():
    """Pin the mixed winding: W/E normals-out, the rest normals-in."""
    quads = unit_cube_quads("block/stone")
    axes = {
        TIER_DOWN: (0, -1, 0),
        TIER_UP: (0, 1, 0),
        TIER_NORTH: (0, 0, -1),
        TIER_SOUTH: (0, 0, 1),
        TIER_WEST: (-1, 0, 0),
        TIER_EAST: (1, 0, 0),
    }
    for quad in quads:
        tier = int(quad["shade"])
        edge1 = quad["vtx"][1] - quad["vtx"][0]
        edge2 = quad["vtx"][2] - quad["vtx"][1]
        dot = float(np.dot(np.cross(edge1, edge2), axes[tier]))
        if tier in (TIER_WEST, TIER_EAST):
            assert dot > 0
        else:
            assert dot < 0


def test_unit_cube_quads_unshaded_uses_up_tier():
    quads = unit_cube_quads("block/stone", shade_faces=False)
    assert (quads["shade"] == TIER_UP).all()
    assert np.array_equal(quads["vtx"], unit_cube_quads("block/stone")["vtx"])


def test_sort_quads_is_canonical():
    quads = unit_cube_quads("t")
    shuffled = quads[[3, 0, 5, 2, 1, 4]]
    first = sort_quads(shuffled)
    second = sort_quads(first)
    assert first.dtype == QUAD_DTYPE
    assert first.tobytes() == second.tobytes()
    keys = [row["vtx"].tobytes() for row in quads]
    expected = sorted(range(6), key=keys.__getitem__)
    assert first["shade"].tolist() == quads["shade"][expected].tolist()
    assert first["vtx"].tolist() == quads["vtx"][expected].tolist()


def test_sort_quads_returns_copy_without_touching_input():
    quads = unit_cube_quads("t")
    before = quads.tobytes()
    out = sort_quads(quads)
    assert out is not quads
    out["shade"][:] = 9
    assert quads.tobytes() == before


def test_sort_quads_handles_empty_input():
    empty = np.empty(0, dtype=QUAD_DTYPE)
    out = sort_quads(empty)
    assert len(out) == 0
    assert out.dtype == QUAD_DTYPE


def test_parallelogram_detection():
    rect = [(0, 0, 0), (16, 0, 0), (16, 0, 16), (0, 0, 16)]
    trap = [(0, 0, 0), (16, 0, 0), (12, 0, 16), (4, 0, 16)]
    skew = [(0, 0, 0), (16, 0, 0), (16, 0, 16), (0, 0, 16.1)]
    quads = np.concatenate([_quad(v) for v in (rect, trap, skew)])
    flags = is_parallelogram(quads)
    assert isinstance(flags, np.ndarray)
    assert flags.dtype == np.dtype("?")
    assert flags.tolist() == [True, False, False]


def test_parallelogram_eps_tolerance():
    near = [(0, 0, 0), (16, 0, 0), (16, 0, 16), (0, 0, 16.002)]
    assert not is_parallelogram(_quad(near)).any()
    assert is_parallelogram(_quad(near), eps=1e-2).all()


def test_parallelogram_unit_cube_faces_all_true():
    assert is_parallelogram(unit_cube_quads("t")).all()


def test_parallelogram_empty_input():
    out = is_parallelogram(np.empty(0, dtype=QUAD_DTYPE))
    assert len(out) == 0
    assert out.dtype == np.dtype("?")


# Cull metadata: every producer marks its quads with a cullface tier or
# CULL_NONE; extraction resolves model JSON cullface through variant rotation.


def test_cull_none_producers():
    assert (unit_cube_quads("block/stone")["cull"] == CULL_NONE).all()
    unshaded = unit_cube_quads("block/stone", shade_faces=False)
    assert (unshaded["cull"] == CULL_NONE).all()


def test_cull_dtype_field():
    assert QUAD_DTYPE.names[-1] == "cull"
    assert QUAD_DTYPE["cull"] == np.dtype("u1")
