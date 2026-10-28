"""World composition contract: air exclusion, culling, yaw, spans."""

import numpy as np
from helpers.litematica_packer import build_litematic_bytes

from litematic import loads
from litematic.render.isometric import (
    SHAPE_CUBE,
    SHAPE_SLAB_BOTTOM,
    SHAPE_SLAB_TOP,
    _compose,
)
from litematic.render.pack import Textures

_FACES = ("up", "east", "south", "down", "north", "west")


def _textures(opaque=(), alpha128=(), slabs=()):
    arrays = {}
    maps = {}

    def register(name, rgba):
        key = f"block/{name.split(':')[-1]}"
        arrays[key] = np.full((16, 16, 4), rgba, dtype=np.uint8)
        maps[f"{name}[]"] = {face: key for face in _FACES}

    for name in opaque:
        register(name, (140, 140, 140, 255))
    for name in alpha128:
        register(name, (160, 208, 224, 128))
    for name in slabs:
        register(name, (110, 84, 52, 255))
    return Textures(version="pack", arrays=arrays, maps=maps, pack_mtime=0.0)


def _schematic(regions):
    return loads(
        build_litematic_bytes(
            [
                {
                    "name": name,
                    "position": position,
                    "size": size,
                    "palette": palette,
                    "indices_yzx": indices,
                }
                for name, position, size, palette, indices in regions
            ]
        )
    )


def _single_block(name, properties=None):
    palette = [("minecraft:air", {}), (name, properties or {})]
    return _schematic(
        [
            (
                "Main",
                (0, 0, 0),
                (1, 1, 1),
                palette,
                np.array([1], dtype=np.uint16),
            )
        ]
    )


def _shell_schematic(shell_name, center_name, shell_props=None, center_props=None):
    palette = [
        ("minecraft:air", {}),
        (shell_name, shell_props or {}),
        (center_name, center_props or {}),
    ]
    indices = np.ones(27, dtype=np.uint16)
    indices[13] = 2  # center of 3x3x3
    return _schematic(
        [("Main", (0, 0, 0), (3, 3, 3), palette, indices)]
    )


def _scene_tuples(scene):
    return sorted(zip(scene.x.tolist(), scene.y.tolist(), scene.z.tolist()))


TEX = _textures(
    opaque=["minecraft:stone"],
    alpha128=["minecraft:glass"],
    slabs=["minecraft:oak_slab"],
)


def test_single_block_present_with_spans():
    scene = _compose(_single_block("minecraft:stone"), 0, TEX)
    assert _scene_tuples(scene) == [(0, 0, 0)]
    assert scene.shape.tolist() == [SHAPE_CUBE]
    assert scene.tex_ok.tolist() == [True]
    assert (scene.span_w, scene.span_h) == (2, 4)


def test_glass_shell_keeps_center_visible():
    scene = _compose(_shell_schematic("minecraft:glass", "minecraft:stone"), 0, TEX)
    assert (1, 1, 1) in _scene_tuples(scene)


def test_stone_shell_culls_center():
    scene = _compose(_shell_schematic("minecraft:stone", "minecraft:stone"), 0, TEX)
    assert (1, 1, 1) not in _scene_tuples(scene)
    assert len(scene.x) == 19  # 27 - 8 blocks with all three camera-side neighbors


def test_magenta_pack_miss_shell_culls_center():
    scene = _compose(
        _shell_schematic("minecraft:gold_block", "minecraft:stone"), 0, TEX
    )
    assert (1, 1, 1) not in _scene_tuples(scene)


def test_pack_miss_center_culled_by_stone_shell():
    scene = _compose(
        _shell_schematic("minecraft:stone", "minecraft:gold_block"), 0, TEX
    )
    assert (1, 1, 1) not in _scene_tuples(scene)
    assert scene.tex_ok.tolist() == [True] * 19


def test_slab_shell_keeps_center_visible_and_records_shapes():
    scene = _compose(
        _shell_schematic("minecraft:oak_slab", "minecraft:stone"), 0, TEX
    )
    assert (1, 1, 1) in _scene_tuples(scene)
    assert scene.shape.tolist().count(SHAPE_SLAB_BOTTOM) == 26
    assert scene.shape.tolist().count(SHAPE_CUBE) == 1


def test_slab_shapes_by_type_property():
    scene = _compose(_single_block("minecraft:oak_slab", {"type": "top"}), 0, TEX)
    assert scene.shape.tolist() == [SHAPE_SLAB_TOP]
    scene = _compose(_single_block("minecraft:oak_slab", {"type": "double"}), 0, TEX)
    assert scene.shape.tolist() == [SHAPE_CUBE]


def test_air_at_nonzero_palette_index_excluded():
    palette = [("minecraft:stone", {}), ("minecraft:air", {})]
    schem = _schematic(
        [
            (
                "Main",
                (0, 0, 0),
                (1, 1, 2),
                palette,
                np.array([0, 1], dtype=np.uint16),
            )
        ]
    )
    scene = _compose(schem, 0, TEX)
    assert _scene_tuples(scene) == [(0, 0, 0)]


def test_multi_region_world_offsets():
    regions = [
        (
            "A",
            (0, 0, 0),
            (1, 1, 1),
            [("minecraft:air", {}), ("minecraft:stone", {})],
            np.array([1], dtype=np.uint16),
        ),
        (
            "B",
            (10, 0, 0),
            (1, 1, 1),
            [("minecraft:air", {}), ("minecraft:stone", {})],
            np.array([1], dtype=np.uint16),
        ),
    ]
    scene = _compose(_schematic(regions), 0, TEX)
    assert _scene_tuples(scene) == [(0, 0, 0), (10, 0, 0)]
    assert (scene.span_w, scene.span_h) == (12, 14)


def _pos_of_shape(scene, shape):
    mask = scene.shape == shape
    return (int(scene.x[mask][0]), int(scene.y[mask][0]), int(scene.z[mask][0]))


def test_yaw_rotation_quadrants():
    palette = [
        ("minecraft:air", {}),
        ("minecraft:stone", {}),
        ("minecraft:oak_slab", {}),
    ]
    indices = np.zeros(9, dtype=np.uint16)
    indices[2] = 1  # stone at (x=2, y=0, z=0) in yzx order
    indices[6] = 2  # slab at (x=0, y=0, z=2)
    schem = _schematic([("Main", (0, 0, 0), (3, 1, 3), palette, indices)])
    expected = {
        0: ((2, 0, 0), (0, 0, 2)),
        1: ((0, 0, 0), (2, 0, 2)),
        2: ((0, 0, 2), (2, 0, 0)),
        3: ((2, 0, 2), (0, 0, 0)),
    }
    for yaw, (stone_pos, slab_pos) in expected.items():
        scene = _compose(schem, yaw, TEX)
        assert _pos_of_shape(scene, SHAPE_CUBE) == stone_pos, yaw
        assert _pos_of_shape(scene, SHAPE_SLAB_BOTTOM) == slab_pos, yaw


def test_all_air_returns_empty_scene():
    palette = [("minecraft:air", {})]
    schem = _schematic(
        [("Main", (0, 0, 0), (2, 2, 2), palette, np.zeros(8, dtype=np.uint16))]
    )
    scene = _compose(schem, 0, TEX)
    assert len(scene.x) == 0
    assert (scene.span_w, scene.span_h) == (2, 4)
