"""Shape template contract: pixel partition, cell bounds, texel LUTs."""

import numpy as np
import pytest

from litematic.parse.model import BlockState
from litematic.render.templates import (
    SHAPE_CUBE,
    SHAPE_SLAB_BOTTOM,
    SHAPE_SLAB_TOP,
    _shape_of,
    _shape_templates,
)

_ALL_SHAPES = [SHAPE_CUBE, SHAPE_SLAB_BOTTOM, SHAPE_SLAB_TOP]


def test_cube_sprite_is_exact_partition():
    for t in (1, 2, 4):
        tm = _shape_templates(t)[SHAPE_CUBE]
        faces = [tm.top, tm.east, tm.south]
        assert sum(len(f) for f in faces) == 12 * t * t
        seen = set()
        for f in faces:
            for dy, dx in f:
                key = (int(dy), int(dx))
                assert key not in seen, "faces must be pairwise disjoint"
                seen.add(key)


def test_slab_sprites_have_full_top_and_half_sides():
    for t in (1, 2, 4):
        tm = _shape_templates(t)[SHAPE_SLAB_BOTTOM]
        assert len(tm.top) == 4 * t * t
        assert len(tm.east) == 2 * t * t
        assert len(tm.south) == 2 * t * t
        tm2 = _shape_templates(t)[SHAPE_SLAB_TOP]
        assert len(tm2.top) == 4 * t * t
        assert len(tm2.east) == 2 * t * t
        assert len(tm2.south) == 2 * t * t


def test_all_shapes_fit_cube_cell_and_slabs_use_3t_rows():
    for t in (1, 2, 4):
        tm = _shape_templates(t)
        for shape in (SHAPE_CUBE, SHAPE_SLAB_BOTTOM, SHAPE_SLAB_TOP):
            all_px = np.concatenate([tm[shape].top, tm[shape].east, tm[shape].south])
            assert all_px[:, 1].min() >= 0
            assert all_px[:, 1].max() < 4 * t
            assert all_px[:, 0].min() >= 0
            assert all_px[:, 0].max() < 4 * t
        for shape, first_row in ((SHAPE_SLAB_TOP, 0), (SHAPE_SLAB_BOTTOM, t)):
            all_px = np.concatenate([tm[shape].top, tm[shape].east, tm[shape].south])
            rows = np.unique(all_px[:, 0])
            assert len(rows) == 3 * t
            assert rows.min() == first_row


def test_t1_cube_rows_match_reference_tile():
    tm = _shape_templates(1)[SHAPE_CUBE]
    rows = np.zeros((4, 4), dtype=np.int8)
    for face in (tm.top, tm.east, tm.south):
        for dy, dx in face:
            rows[int(dy), int(dx)] = 1
    assert rows.tolist() == [
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
    ]


def test_texel_coords_within_bounds():
    for t in (1, 2, 4):
        tm = _shape_templates(t)[SHAPE_CUBE]
        for lut in (tm.texel_top, tm.texel_east, tm.texel_south):
            assert lut.min() >= 0
            assert lut.max() < 16


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (BlockState("minecraft:stone"), SHAPE_CUBE),
        (BlockState("minecraft:oak_stairs", {"facing": "north"}), SHAPE_CUBE),
        (BlockState("minecraft:oak_slab"), SHAPE_SLAB_BOTTOM),
        (BlockState("minecraft:oak_slab", {"type": "bottom"}), SHAPE_SLAB_BOTTOM),
        (BlockState("minecraft:oak_slab", {"type": "top"}), SHAPE_SLAB_TOP),
        (BlockState("minecraft:oak_slab", {"type": "double"}), SHAPE_CUBE),
    ],
)
def test_shape_of(state, expected):
    assert _shape_of(state) == expected


def test_slab_side_texels_cropped_not_squashed():
    for t in (1, 2, 4):
        tm = _shape_templates(t)
        cube = tm[SHAPE_CUBE]
        cube_y = np.concatenate([cube.texel_east[:, 1], cube.texel_south[:, 1]])
        assert cube_y.min() < 8
        assert cube_y.max() >= 8
        bottom = tm[SHAPE_SLAB_BOTTOM]
        bottom_y = np.concatenate([bottom.texel_east[:, 1], bottom.texel_south[:, 1]])
        assert bottom_y.min() >= 8
        top = tm[SHAPE_SLAB_TOP]
        top_y = np.concatenate([top.texel_east[:, 1], top.texel_south[:, 1]])
        assert top_y.max() < 8
