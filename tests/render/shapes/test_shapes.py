"""Builtin shape table contract: family geometry, dispatch, placeholders."""

import numpy as np
import pytest

from litematic.render.errors import RenderError
from litematic.render.quads import (
    QUAD_DTYPE,
    TIER_DOWN,
    TIER_EAST,
    TIER_UP,
    TIER_WEST,
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


def test_box_auto_uv_west_east_follow_vanilla_axes():
    quads = box_quads(2, 4, 6, 14, 12, 16)
    for quad in quads:
        tier = int(quad["shade"])
        if tier not in (TIER_WEST, TIER_EAST):
            continue
        for (x, y, z), (u, v) in zip(
            quad["vtx"].tolist(), quad["uv"].tolist()
        ):
            if tier == TIER_WEST:
                assert (u, v) == (z, 16.0 - y)
            else:
                assert (u, v) == (16.0 - z, 16.0 - y)


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


def test_fence_post_and_arms_follow_properties():
    lone = builtin_quads("oak_fence", {})
    assert _extent(lone) == ((6, 0, 6), (10, 16, 10))
    assert len(lone) == 6
    north = builtin_quads("oak_fence", {"north": "true"})
    assert len(north) == 18
    assert _span(north, 0) == (6.0, 10.0)
    assert _span(north, 1) == (0.0, 16.0)
    assert _span(north, 2) == (0.0, 10.0)
    rails = north[north["vtx"][:, :, 2].min(axis=1) == 0.0]
    assert sorted(set(rails["vtx"][:, :, 1].ravel().tolist())) == [
        6.0,
        9.0,
        12.0,
        15.0,
    ]
    all_sides = builtin_quads(
        "oak_fence",
        {"north": "true", "south": "true", "east": "true", "west": "true"},
    )
    assert _extent(all_sides) == ((0, 0, 0), (16, 16, 16))
    assert len(all_sides) == 54


def test_fence_gate_bar_swings_when_open():
    closed_north = builtin_quads("oak_fence_gate", {"facing": "north"})
    assert len(closed_north) == 12
    assert _extent(closed_north) == ((0, 0, 6), (16, 16, 10))
    open_north = builtin_quads(
        "oak_fence_gate", {"facing": "north", "open": "true"}
    )
    assert _extent(open_north) == ((6, 0, 0), (10, 16, 16))
    closed_east = builtin_quads("oak_fence_gate", {"facing": "east"})
    assert _extent(closed_east) == ((6, 0, 0), (10, 16, 16))
    open_east = builtin_quads(
        "oak_fence_gate", {"facing": "east", "open": "true"}
    )
    assert _extent(open_east) == ((0, 0, 6), (16, 16, 10))


def test_wall_post_height_and_arms():
    lone = builtin_quads("cobblestone_wall", {})
    assert _extent(lone) == ((4, 0, 4), (12, 14, 12))
    up = builtin_quads("cobblestone_wall", {"up": "true"})
    assert _extent(up) == ((4, 0, 4), (12, 16, 12))
    north = builtin_quads("cobblestone_wall", {"north": "low", "up": "true"})
    assert len(north) == 12
    assert _span(north, 0) == (4.0, 12.0)
    assert _span(north, 2) == (0.0, 12.0)
    both = builtin_quads("cobblestone_wall", {"north": "tall", "south": "low"})
    assert _span(both, 2) == (0.0, 16.0)
    assert len(both) == 18
    none_side = builtin_quads("cobblestone_wall", {"east": "none"})
    assert _extent(none_side) == ((4, 0, 4), (12, 14, 12))


def test_pane_unconnected_is_full_bar():
    pane = builtin_quads("glass_pane", {})
    assert _extent(pane) == ((7, 0, 0), (9, 16, 16))
    assert len(pane) == 12


def test_pane_arms_follow_connections():
    north = builtin_quads("glass_pane", {"north": "true"})
    assert _extent(north) == ((7, 0, 0), (9, 16, 9))
    assert len(north) == 12
    cross = builtin_quads(
        "glass_pane",
        {"north": "true", "south": "true", "east": "true", "west": "true"},
    )
    assert _extent(cross) == ((0, 0, 0), (16, 16, 16))
    assert len(cross) == 30


def test_iron_bars_share_pane_geometry():
    bars = builtin_quads("iron_bars", {"east": "true"})
    assert _extent(bars) == ((7, 0, 7), (16, 16, 9))
    assert len(bars) == 12


def test_cross_plant_two_double_sided_quads():
    quads = builtin_quads("poppy", {})
    assert quads.dtype == QUAD_DTYPE
    assert len(quads) == 2
    assert quads["dbl"].all()
    assert (quads["texkey"] == "").all()
    assert (quads["shade"] == TIER_UP).all()
    assert _span(quads, 1) == (0.0, 16.0)
    for quad in quads["vtx"]:
        bottom = quad[:2]
        along_xz = np.allclose(bottom[:, 0], bottom[:, 2])
        anti_diagonal = np.allclose(bottom[:, 0] + bottom[:, 2], 16.0)
        assert along_xz or anti_diagonal


def test_cross_plant_name_list_matches():
    names = (
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
        "oak_sapling",
        "mangrove_propagule",
        "minecraft:poppy",
    )
    for name in names:
        quads = builtin_quads(name, {})
        assert quads is not None, name
        assert len(quads) == 2, name
        assert quads["dbl"].all(), name


def test_multi_box_families_stay_parallelograms():
    stairs = builtin_quads("oak_stairs", {"facing": "east", "half": "bottom"})
    lectern = builtin_quads("lectern", {})
    fence = builtin_quads("oak_fence", {"north": "true"})
    gate = builtin_quads(
        "oak_fence_gate", {"facing": "north", "open": "true"}
    )
    wall = builtin_quads("cobblestone_wall", {"north": "low"})
    pane = builtin_quads("glass_pane", {"east": "true"})
    plant = builtin_quads("poppy", {})
    for quads in (stairs, lectern, fence, gate, wall, pane, plant):
        assert is_parallelogram(quads).all()


def test_redstone_wire_unconnected_is_single_dot_plane():
    q = builtin_quads("minecraft:redstone_wire", {"power": "0"})
    assert len(q) == 1
    assert q["shade"][0] == TIER_UP
    assert not q["dbl"][0]
    assert q["texkey"][0] == "block/redstone_dust_dot@76,0,0"
    assert q["vtx"][0][:, 1].min() == 0.0
    assert q["vtx"][0][:, 1].max() == 0.0


def test_redstone_wire_arms_uv_orientation_and_power_tint():
    q = builtin_quads(
        "minecraft:redstone_wire",
        {"north": "side", "east": "up", "power": "15"},
    )
    assert len(q) == 3
    assert set(q["texkey"]) == {"block/redstone_dust_line0@255,26,26"}
    ground = q[~q["dbl"]]
    assert len(ground) == 2
    # North-south planes sample the line column along x; east-west planes
    # transpose the uv so the same texture runs along z.
    assert any(bool(np.allclose(g["uv"][:, 0], g["vtx"][:, 0])) for g in ground)
    assert any(bool(np.allclose(g["uv"][:, 0], g["vtx"][:, 2])) for g in ground)
    strip = q[q["dbl"]]
    assert len(strip) == 1
    assert strip["vtx"][0][:, 0].min() == 16.0
    assert strip["vtx"][0][:, 0].max() == 16.0
    assert strip["vtx"][0][:, 1].min() == 0.0
    assert strip["vtx"][0][:, 1].max() == 16.0


def test_redstone_wire_malformed_power_falls_to_zero():
    q = builtin_quads("minecraft:redstone_wire", {"power": "x"})
    assert q["texkey"][0] == "block/redstone_dust_dot@76,0,0"


def test_composter_full_cube():
    q = builtin_quads("minecraft:composter", {"level": "3"})
    assert len(q) == 6
    assert q["vtx"][..., 0].min() == 0.0
    assert q["vtx"][..., 0].max() == 16.0
    assert q["vtx"][..., 1].min() == 0.0
    assert q["vtx"][..., 1].max() == 16.0
    assert q["vtx"][..., 2].min() == 0.0
    assert q["vtx"][..., 2].max() == 16.0
