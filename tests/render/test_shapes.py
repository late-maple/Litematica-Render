"""Builtin shape table contract: family geometry, dispatch, placeholders."""

import numpy as np
import pytest

from litematic.render.errors import RenderError
from litematic.render.quads import (
    QUAD_DTYPE,
    TIER_DOWN,
    TIER_UP,
    is_parallelogram,
    unit_cube_quads,
)
from litematic.render.shapes import box_quads, builtin_quads


def _extent(quads):
    """Return the (min, max) vertex corner across all quads per axis."""
    vtx = quads["vtx"].reshape(-1, 3)
    return tuple(vtx.min(axis=0)), tuple(vtx.max(axis=0))


def _span(quads, axis):
    """Return the (min, max) vertex coordinate along one axis."""
    values = quads["vtx"][..., axis]
    return float(values.min()), float(values.max())


def _tier(quads, shade):
    """Return the subset of quads whose face tier equals shade."""
    return quads[quads["shade"] == shade]


def test_box_quads_full_cube_matches_quad_core():
    cube = box_quads(0, 0, 0, 16, 16, 16)
    reference = unit_cube_quads("")
    assert cube.dtype == QUAD_DTYPE
    assert len(cube) == 6
    for tier in range(6):
        row = cube[cube["shade"] == tier][0]
        ref = reference[reference["shade"] == tier][0]
        assert np.array_equal(row["vtx"], ref["vtx"])
        assert np.array_equal(row["uv"], ref["uv"])
    assert (cube["texkey"] == "").all()
    assert not cube["dbl"].any()


def test_box_quads_auto_uv_tracks_box_extent():
    half = box_quads(0, 0, 0, 16, 8, 16)
    south = half[half["shade"] == 3][0]
    assert np.array_equal(
        south["uv"], np.array([(0, 8), (16, 8), (16, 16), (0, 16)], dtype="f4")
    )


def test_slab_types_place_the_half_box():
    bottom = builtin_quads("oak_slab", {})
    assert _extent(bottom) == ((0, 0, 0), (16, 8, 16))
    top = builtin_quads("oak_slab", {"type": "top"})
    assert _extent(top) == ((0, 8, 0), (16, 16, 16))
    double = builtin_quads("oak_slab", {"type": "double"})
    assert _extent(double) == ((0, 0, 0), (16, 16, 16))
    assert len(double) == 6


def test_stairs_two_boxes_follow_facing():
    quads = builtin_quads("oak_stairs", {"facing": "east", "half": "bottom"})
    assert len(quads) == 12
    assert sorted(quads["shade"].tolist()) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert _extent(quads) == ((0, 0, 0), (16, 16, 16))
    base, step = sorted(_tier(quads, TIER_UP), key=lambda q: q["vtx"][0][1])
    assert float(base["vtx"][0][1]) == 8.0
    assert _span(step, 0) == (8.0, 16.0)
    assert _span(step, 1) == (16.0, 16.0)


def test_stairs_top_half_flips_step_below():
    quads = builtin_quads("oak_stairs", {"facing": "north", "half": "top"})
    low, high = sorted(_tier(quads, TIER_DOWN), key=lambda q: q["vtx"][0][1])
    assert float(low["vtx"][0][1]) == 0.0
    assert _span(low, 2) == (0.0, 8.0)
    assert float(high["vtx"][0][1]) == 8.0


def test_snow_layers_set_height():
    assert _span(builtin_quads("snow", {"layers": "3"}), 1) == (0.0, 6.0)
    assert _span(builtin_quads("snow", {"layers": "8"}), 1) == (0.0, 16.0)
    assert _span(builtin_quads("snow", {}), 1) == (0.0, 2.0)
    assert _span(builtin_quads("snow", {"layers": "0"}), 1) == (0.0, 2.0)
    assert _span(builtin_quads("snow", {"layers": "99"}), 1) == (0.0, 16.0)


def test_water_level_drops_height():
    assert _span(builtin_quads("water", {"level": "3"}), 1) == (0.0, 13.0)
    assert _span(builtin_quads("water", {}), 1) == (0.0, 16.0)
    assert _span(builtin_quads("water", {"level": "15"}), 1) == (0.0, 1.0)
    assert _span(builtin_quads("water", {"level": "high"}), 1) == (0.0, 16.0)


def test_fluid_siblings_share_water_geometry():
    assert _span(builtin_quads("lava", {"level": "2"}), 1) == (0.0, 14.0)
    assert _span(builtin_quads("bubble_column", {"level": "5"}), 1) == (0.0, 11.0)


def test_carpet_is_one_pixel_thick():
    for name in ("red_carpet", "moss_carpet"):
        quads = builtin_quads(name, {})
        assert _extent(quads) == ((0, 0, 0), (16, 1, 16))
        assert len(quads) == 6


def test_rail_family_is_one_pixel_thick():
    for name in ("rail", "powered_rail", "detector_rail", "activator_rail"):
        quads = builtin_quads(name, {})
        assert _extent(quads) == ((0, 0, 0), (16, 1, 16))
        assert len(quads) == 6


def test_pressure_plate_footprint():
    quads = builtin_quads("oak_pressure_plate", {})
    assert _extent(quads) == ((1, 0, 1), (15, 1, 15))


def test_repeater_and_comparator_are_low_panels():
    for name in ("repeater", "comparator"):
        quads = builtin_quads(name, {})
        assert _extent(quads) == ((2, 0, 2), (14, 2, 14))


def test_trapdoor_open_moves_to_edge():
    quads = builtin_quads("oak_trapdoor", {"open": "true", "facing": "south"})
    assert _span(quads, 0) == (0.0, 16.0)
    assert _span(quads, 1) == (0.0, 16.0)
    assert _span(quads, 2) == (13.0, 16.0)
    east = builtin_quads("oak_trapdoor", {"open": "true", "facing": "east"})
    assert _span(east, 0) == (13.0, 16.0)
    west = builtin_quads("oak_trapdoor", {"open": "true", "facing": "west"})
    assert _span(west, 0) == (0.0, 3.0)
    north = builtin_quads("oak_trapdoor", {"open": "true", "facing": "north"})
    assert _span(north, 2) == (0.0, 3.0)


def test_trapdoor_closed_follows_half():
    assert _span(builtin_quads("oak_trapdoor", {}), 1) == (0.0, 3.0)
    top = builtin_quads("oak_trapdoor", {"half": "top"})
    assert _span(top, 1) == (13.0, 16.0)


def test_door_swing_follows_open_and_facing():
    closed_south = builtin_quads("oak_door", {"facing": "south"})
    assert _span(closed_south, 2) == (13.0, 16.0)
    open_south = builtin_quads("oak_door", {"facing": "south", "open": "true"})
    assert _span(open_south, 0) == (13.0, 16.0)
    closed_north = builtin_quads("oak_door", {})
    assert _span(closed_north, 2) == (0.0, 3.0)
    open_east = builtin_quads("oak_door", {"facing": "east", "open": "true"})
    assert _span(open_east, 2) == (0.0, 3.0)


def test_ladder_hugs_facing_wall():
    assert _span(builtin_quads("ladder", {"facing": "east"}), 0) == (14.0, 16.0)
    assert _span(builtin_quads("ladder", {"facing": "south"}), 2) == (14.0, 16.0)
    assert _span(builtin_quads("ladder", {}), 2) == (0.0, 2.0)


def test_button_face_positions():
    floor = builtin_quads("oak_button", {})
    assert _extent(floor) == ((5, 0, 5), (11, 2, 11))
    ceiling = builtin_quads("oak_button", {"face": "ceiling"})
    assert _span(ceiling, 1) == (14.0, 16.0)
    wall_south = builtin_quads("oak_button", {"face": "wall", "facing": "south"})
    assert _extent(wall_south) == ((5, 5, 14), (11, 11, 16))
    wall_north = builtin_quads("oak_button", {"face": "wall"})
    assert _span(wall_north, 2) == (0.0, 2.0)


def test_sign_and_wall_sign_share_post_geometry():
    for name in ("oak_sign", "oak_wall_sign"):
        quads = builtin_quads(name, {})
        assert _extent(quads) == ((7, 0, 0), (9, 16, 16))


def test_banner_and_wall_banner_share_post_geometry():
    for name in ("oak_banner", "oak_wall_banner"):
        quads = builtin_quads(name, {})
        assert _extent(quads) == ((7, 0, 0), (9, 16, 16))


def test_chest_family_is_inset_box():
    for name in ("chest", "trapped_chest", "ender_chest"):
        quads = builtin_quads(name, {})
        assert _extent(quads) == ((1, 0, 1), (15, 14, 15))


def test_shulker_box_and_barrel_keep_full_height():
    for name in ("shulker_box", "white_shulker_box", "barrel"):
        quads = builtin_quads(name, {})
        assert _extent(quads) == ((1, 0, 1), (15, 16, 15))


def test_head_and_skull_box():
    for name in ("player_head", "skeleton_skull"):
        quads = builtin_quads(name, {})
        assert _extent(quads) == ((4, 0, 4), (12, 8, 12))


def test_cactus_is_inset_column():
    quads = builtin_quads("cactus", {})
    assert _extent(quads) == ((1, 0, 1), (15, 16, 15))


def test_lily_pad_is_one_pixel_thick():
    quads = builtin_quads("lily_pad", {})
    assert _extent(quads) == ((0, 0, 0), (16, 1, 16))


def test_conduit_is_centered_core():
    quads = builtin_quads("conduit", {})
    assert _extent(quads) == ((4, 4, 4), (12, 12, 12))


def test_lectern_is_base_plus_desk():
    quads = builtin_quads("lectern", {})
    assert len(quads) == 12
    assert _span(quads, 0) == (0.0, 16.0)
    low, high = sorted(_tier(quads, TIER_UP), key=lambda q: q["vtx"][0][1])
    assert float(low["vtx"][0][1]) == 4.0
    assert _span(low, 0) == (0.0, 16.0)
    assert float(high["vtx"][0][1]) == 12.0
    assert _span(high, 0) == (4.0, 12.0)


def test_torch_standing_is_small_post():
    quads = builtin_quads("torch", {})
    assert _extent(quads) == ((7, 0, 7), (9, 10, 9))


def test_wall_torch_lies_against_facing_wall():
    south = builtin_quads("wall_torch", {"facing": "south"})
    assert _extent(south) == ((7, 3, 14), (9, 13, 16))
    west = builtin_quads("wall_torch", {"facing": "west"})
    assert _span(west, 0) == (0.0, 2.0)
    assert _span(west, 1) == (3.0, 13.0)


def test_unknown_name_returns_none():
    assert builtin_quads("stone", {}) is None
    assert builtin_quads("minecraft:dirt", {}) is None
    assert builtin_quads("furnace", {}) is None


def test_empty_name_is_unmatched_and_returns_none():
    assert builtin_quads("", {}) is None
    assert builtin_quads("minecraft:", {}) is None


def test_namespace_prefix_is_stripped():
    quads = builtin_quads("minecraft:water", {"level": "5"})
    assert _span(quads, 1) == (0.0, 11.0)


def test_builtin_quads_use_placeholder_texkeys():
    quads = builtin_quads("chest", {})
    assert quads.dtype == QUAD_DTYPE
    assert (quads["texkey"] == "").all()
    assert not quads["dbl"].any()
    assert sorted(quads["shade"].tolist()) == [0, 1, 2, 3, 4, 5]


def test_invalid_property_values_fall_back_to_defaults():
    ladder = builtin_quads("ladder", {"facing": "sideways"})
    assert _span(ladder, 2) == (0.0, 2.0)
    assert _span(builtin_quads("snow", {"layers": "many"}), 1) == (0.0, 2.0)
    assert _span(builtin_quads("water", {"level": ""}), 1) == (0.0, 16.0)


def test_contract_violations_raise_render_error():
    with pytest.raises(RenderError):
        builtin_quads(123, {})
    with pytest.raises(RenderError):
        builtin_quads("water", None)
    with pytest.raises(RenderError):
        builtin_quads("water", "level=3")


def test_multi_box_families_stay_parallelograms():
    stairs = builtin_quads("oak_stairs", {"facing": "east", "half": "bottom"})
    lectern = builtin_quads("lectern", {})
    assert is_parallelogram(stairs).all()
    assert is_parallelogram(lectern).all()
