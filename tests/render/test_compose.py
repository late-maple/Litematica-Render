"""Scene composition contract: quad expansion, yaw remap, culling, AO."""

import json
import zipfile

import numpy as np
from helpers.litematica_packer import build_litematic_bytes

from litematic import loads
from litematic.render.isometric import (
    _MAGENTA_TEXKEY,
    _backfill_builtin,
    _compose,
)
from litematic.render.pack import Textures, load_pack
from litematic.render.quads import (
    QUAD_DTYPE,
    TIER_EAST,
    TIER_SOUTH,
    TIER_UP,
    unit_cube_quads,
)
from litematic.render.shapes import builtin_quads

_STONE = (140, 140, 140, 255)


def _textures(occl):
    """Stone cube pack whose opaque flag is the given boolean."""
    return Textures(
        version="pack-quads",
        arrays={"block/stone": np.full((16, 16, 4), _STONE, np.uint8)},
        quads_map={"minecraft:stone[]": unit_cube_quads("block/stone")},
        occl_map={"minecraft:stone[]": occl},
        pack_mtime=0.0,
    )


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


def _shell_schematic(shell_name, center_name):
    palette = [("minecraft:air", {}), (shell_name, {}), (center_name, {})]
    indices = np.ones(27, dtype=np.uint16)
    indices[13] = 2  # center of 3x3x3
    return _schematic([("Main", (0, 0, 0), (3, 3, 3), palette, indices)])


def _cell_schematic(cells):
    """One region holding stone at the given (x, y, z) cells."""
    sx = max(cell[0] for cell in cells) + 1
    sy = max(cell[1] for cell in cells) + 1
    sz = max(cell[2] for cell in cells) + 1
    indices = np.zeros(sx * sy * sz, dtype=np.uint16)
    for x, y, z in cells:
        indices[(y * sz + z) * sx + x] = 1
    palette = [("minecraft:air", {}), ("minecraft:stone", {})]
    return _schematic([("Main", (0, 0, 0), (sx, sy, sz), palette, indices)])


def _scene_tuples(scene):
    return sorted(set(zip(scene.x.tolist(), scene.y.tolist(), scene.z.tolist())))


def _up_ao(scene, x, y, z):
    """The top-face quad's four AO multipliers at one block cell."""
    mask = (
        (scene.x == x)
        & (scene.y == y)
        & (scene.z == z)
        & (scene.quad_shade == TIER_UP)
    )
    rows = scene.quad_ao[mask]
    assert len(rows) == 1
    return rows[0]


def test_scene_expands_states_to_quads():
    palette = [
        ("minecraft:air", {}),
        ("minecraft:stone", {}),
        ("minecraft:gold_block", {}),
    ]
    indices = np.array([1, 2], dtype=np.uint16)
    schem = _schematic([("Main", (0, 0, 0), (2, 1, 1), palette, indices)])
    scene = _compose(schem, 0, _textures(True))
    assert len(scene.quad_shade) == 6  # 3 camera-facing faces per block
    assert (scene.span_w, scene.span_h) == (3, 5)
    table = list(scene.texkeys)
    assert set(table) == {"block/stone", _MAGENTA_TEXKEY}
    ids = scene.texkey_ids.tolist()
    assert ids.count(table.index("block/stone")) == 3
    assert ids.count(table.index(_MAGENTA_TEXKEY)) == 3
    assert scene.texkeys[_MAGENTA_TEXKEY].shape == (16, 16, 4)


def test_yaw_remaps_tier_and_vertices():
    quads = np.empty(1, QUAD_DTYPE)
    quads["vtx"] = [[(0, 16, 16), (16, 16, 16), (16, 0, 16), (0, 0, 16)]]
    quads["uv"] = [[(1, 2), (15, 2), (15, 14), (1, 14)]]
    quads["texkey"] = "block/stone"
    quads["shade"] = TIER_SOUTH
    quads["dbl"] = False
    tex = Textures(
        version="pack-quads",
        arrays={"block/stone": np.zeros((16, 16, 4), np.uint8)},
        quads_map={"minecraft:stone[]": quads},
        occl_map={"minecraft:stone[]": False},
        pack_mtime=0.0,
    )
    scene = _compose(_single_block("minecraft:stone"), 1, tex)
    assert scene.quad_shade.tolist() == [TIER_EAST]
    assert scene.quad_uv.tolist() == [[[1, 2], [15, 2], [15, 14], [1, 14]]]
    assert scene.quad_vtx.tolist() == [
        [[16, 16, 16], [16, 16, 0], [16, 0, 0], [16, 0, 16]]
    ]


def test_camera_face_filter_keeps_three_tiers():
    scene = _compose(_single_block("minecraft:stone"), 0, _textures(True))
    assert sorted(scene.quad_shade.tolist()) == [TIER_UP, TIER_SOUTH, TIER_EAST]
    assert scene.quad_dbl.tolist() == [False, False, False]


def test_ao_wall_base_darker_than_center():
    cells = [(x, 0, z) for x in range(3) for z in range(3)]
    cells += [(x, 1, 0) for x in range(3)]
    scene = _compose(_cell_schematic(cells), 0, _textures(True))
    near = _up_ao(scene, 1, 0, 1)
    assert (near == np.array([0.6, 0.6, 1.0, 1.0], np.float32)).all()
    far = _up_ao(scene, 1, 0, 2)
    assert (far == np.float32(1.0)).all()


def test_ao_region_border_stays_bright():
    palette = [("minecraft:air", {}), ("minecraft:stone", {})]
    split = _schematic(
        [
            ("A", (0, 0, 0), (1, 1, 1), palette, np.array([1], np.uint16)),
            ("B", (1, 0, 0), (1, 2, 2), palette, np.array([0, 0, 0, 1], np.uint16)),
        ]
    )
    bright = _up_ao(_compose(split, 0, _textures(True)), 0, 0, 0)
    assert (bright == np.float32(1.0)).all()
    joined = _cell_schematic([(0, 0, 0), (1, 1, 1)])
    dark = _up_ao(_compose(joined, 0, _textures(True)), 0, 0, 0)
    assert (dark < np.float32(1.0)).any()


def test_occl_read_from_pack_flag():
    shell = _shell_schematic("minecraft:stone", "minecraft:stone")
    culled = _compose(shell, 0, _textures(True))
    kept = _compose(shell, 0, _textures(False))
    assert (1, 1, 1) not in _scene_tuples(culled)
    assert (1, 1, 1) in _scene_tuples(kept)


def test_culling_dirs_follow_yaw():
    cells = [(x, y, z) for x in range(2) for y in range(2) for z in range(2)]
    schem = _cell_schematic(cells)
    full = sorted(cells)
    for yaw in range(4):
        scene = _compose(schem, yaw, _textures(True))
        seen = _scene_tuples(scene)
        assert (0, 0, 0) not in seen, yaw
        assert seen == [cell for cell in full if cell != (0, 0, 0)], yaw


def test_air_at_nonzero_palette_index_excluded():
    schem = _schematic(
        [
            (
                "Main",
                (0, 0, 0),
                (1, 1, 2),
                [("minecraft:stone", {}), ("minecraft:air", {})],
                np.array([0, 1], dtype=np.uint16),
            )
        ]
    )
    scene = _compose(schem, 0, _textures(True))
    assert _scene_tuples(scene) == [(0, 0, 0)]


def test_yaw_rotation_quadrants():
    tex = Textures(
        version="pack-quads",
        arrays={
            "block/stone": np.full((16, 16, 4), _STONE, np.uint8),
            "block/gold_block": np.full((16, 16, 4), (0, 0, 255, 255), np.uint8),
        },
        quads_map={
            "minecraft:stone[]": unit_cube_quads("block/stone"),
            "minecraft:gold_block[]": unit_cube_quads("block/gold_block"),
        },
        occl_map={"minecraft:stone[]": True, "minecraft:gold_block[]": True},
        pack_mtime=0.0,
    )
    palette = [
        ("minecraft:air", {}),
        ("minecraft:stone", {}),
        ("minecraft:gold_block", {}),
    ]
    indices = np.zeros(9, dtype=np.uint16)
    indices[2] = 1  # stone at (x=2, y=0, z=0) in yzx order
    indices[6] = 2  # gold at (x=0, y=0, z=2)
    schem = _schematic([("Main", (0, 0, 0), (3, 1, 3), palette, indices)])
    expected = {
        0: ((2, 0, 0), (0, 0, 2)),
        1: ((0, 0, 0), (2, 0, 2)),
        2: ((0, 0, 2), (2, 0, 0)),
        3: ((2, 0, 2), (0, 0, 0)),
    }
    for yaw, (stone_pos, gold_pos) in expected.items():
        scene = _compose(schem, yaw, tex)
        table = list(scene.texkeys)
        assert _cells_of(scene, table.index("block/stone")) == [stone_pos], yaw
        assert _cells_of(scene, table.index("block/gold_block")) == [
            gold_pos
        ], yaw


def _cells_of(scene, texkey_id):
    """Sorted unique block cells holding quads of one texkey id."""
    mask = scene.texkey_ids == texkey_id
    return sorted(
        set(zip(scene.x[mask].tolist(), scene.y[mask].tolist(),
                scene.z[mask].tolist()))
    )


def test_builtin_uses_block_named_texture():
    tex = _textures(True)
    tex.arrays["block/oak_slab"] = np.full(
        (16, 16, 4), (110, 84, 52, 255), np.uint8
    )
    scene = _compose(_single_block("minecraft:oak_slab"), 0, tex)
    assert list(scene.texkeys) == ["block/oak_slab"]
    assert len(scene.quad_shade) == 3
    assert scene.quad_dbl.tolist() == [False, False, False]


def test_builtin_slab_uses_planks_material_convention():
    tex = _textures(True)
    tex.arrays["block/oak_planks"] = np.full(
        (16, 16, 4), (156, 127, 78, 255), np.uint8
    )
    scene = _compose(_single_block("minecraft:oak_slab"), 0, tex)
    assert list(scene.texkeys) == ["block/oak_planks"]
    assert len(scene.quad_shade) == 3


def test_multi_region_offsets_and_spans():
    palette = [("minecraft:air", {}), ("minecraft:stone", {})]
    regions = [
        ("A", (0, 0, 0), (1, 1, 1), palette, np.array([1], dtype=np.uint16)),
        ("B", (10, 0, 0), (1, 1, 1), palette, np.array([1], dtype=np.uint16)),
    ]
    scene = _compose(_schematic(regions), 0, _textures(True))
    assert _scene_tuples(scene) == [(0, 0, 0), (10, 0, 0)]
    assert (scene.span_w, scene.span_h) == (12, 14)


def test_all_air_returns_empty_scene():
    schem = _schematic(
        [
            (
                "Main",
                (0, 0, 0),
                (2, 2, 2),
                [("minecraft:air", {})],
                np.zeros(8, dtype=np.uint16),
            )
        ]
    )
    scene = _compose(schem, 0, _textures(True))
    assert len(scene.x) == 0
    assert len(scene.quad_shade) == 0
    assert (scene.span_w, scene.span_h) == (2, 4)
    assert scene.texkeys == {}


def test_dbl_quads_keep_full_ao_next_to_occluders():
    """Spec 3: double-sided quads never take vertex AO."""
    plant = np.empty(2, QUAD_DTYPE)
    plant["vtx"] = [
        [(0, 0, 0), (16, 0, 16), (16, 16, 16), (0, 16, 0)],
        [(16, 0, 0), (0, 0, 16), (0, 16, 16), (16, 16, 0)],
    ]
    plant["uv"] = [[(0, 0), (16, 0), (16, 16), (0, 16)]] * 2
    plant["texkey"] = "block/stone"
    plant["shade"] = TIER_UP
    plant["dbl"] = True
    tex = Textures(
        version="pack-quads",
        arrays={"block/stone": np.full((16, 16, 4), _STONE, np.uint8)},
        quads_map={
            "minecraft:short_grass[]": plant,
            "minecraft:stone[]": unit_cube_quads("block/stone"),
        },
        occl_map={"minecraft:short_grass[]": False, "minecraft:stone[]": True},
        pack_mtime=0.0,
    )
    palette = [
        ("minecraft:air", {}),
        ("minecraft:stone", {}),
        ("minecraft:short_grass", {}),
    ]
    indices = np.ones(9, dtype=np.uint16)
    indices[4] = 2  # plant in the middle of a 3x3 stone floor
    schem = _schematic([("Main", (0, 0, 0), (3, 1, 3), palette, indices)])
    scene = _compose(schem, 0, tex)
    cell = (scene.x == 1) & (scene.y == 0) & (scene.z == 1)
    assert cell.any()
    assert scene.quad_dbl[cell].all()
    assert (scene.quad_ao[cell] == np.float32(1.0)).all()


# Round-2 layering: real packs reach compose's state-aware builtin with a
# texture extraction registered under the canonical block key.


def _fence_pack(path):
    """Vanilla-named oak fence: conditional multipart over planks models."""
    model = json.dumps(
        {
            "textures": {"planks": "block/oak_planks"},
            "elements": [
                {
                    "from": [0, 0, 0],
                    "to": [16, 16, 16],
                    "faces": {
                        "north": {"texture": "#planks"},
                        "south": {"texture": "#planks"},
                        "east": {"texture": "#planks"},
                        "west": {"texture": "#planks"},
                    },
                }
            ],
        }
    )
    entries = {
        "assets/minecraft/blockstates/oak_fence.json": json.dumps(
            {
                "multipart": [
                    {"apply": {"model": "block/fence_post"}},
                    {"when": {"north": "true"}, "apply": {"model": "block/fence_side"}},
                    {"when": {"east": "true"}, "apply": {"model": "block/fence_side"}},
                ]
            }
        ),
        "assets/minecraft/models/block/fence_post.json": model,
        "assets/minecraft/models/block/fence_side.json": model,
        "assets/minecraft/textures/block/oak_planks.png": _png_bytes(
            (168, 90, 50)
        ),
    }
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _png_bytes(color):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (16, 16), (*color, 255)).save(buf, "PNG")
    return buf.getvalue()


def test_real_pack_fence_connection_geometry(tmp_path):
    textures = load_pack(_fence_pack(tmp_path / "fence.zip"))
    alone = _compose(
        _single_block(
            "minecraft:oak_fence",
            {
                "north": "false",
                "south": "false",
                "east": "false",
                "west": "false",
                "waterlogged": "false",
            },
        ),
        0,
        textures,
    )
    linked = _compose(
        _single_block(
            "minecraft:oak_fence",
            {
                "north": "true",
                "south": "false",
                "east": "true",
                "west": "false",
                "waterlogged": "false",
            },
        ),
        0,
        textures,
    )
    assert len(alone.quad_shade) == 2  # post model, camera-facing sides
    # north=true key merges the post and side models.
    assert len(linked.quad_shade) == 6  # post + north side + east side
    assert _MAGENTA_TEXKEY not in alone.texkeys
    assert _MAGENTA_TEXKEY not in linked.texkeys


def test_real_pack_pure_texture_cross_plant_tinted(tmp_path):
    from litematic.render.pack import load_pack

    with zipfile.ZipFile(tmp_path / "grass.zip", "w") as zf:
        zf.writestr(
            "assets/minecraft/textures/block/short_grass.png",
            _png_bytes((98, 158, 68)),
        )
    textures = load_pack(tmp_path / "grass.zip")
    scene = _compose(_single_block("minecraft:short_grass"), 0, textures)
    assert len(scene.quad_shade) == 2
    assert scene.quad_dbl.all()
    assert list(scene.texkeys) == ["block/short_grass@145,189,89"]


# Vanilla base-material texture conventions for model-less packs
# (escalation ruling: supersedes the Task 6 no-family-mapping exclusion).
# Rules fire only when no pack model quads and no same-name texture exist.


def _texture_only_pack(path, textures):
    with zipfile.ZipFile(path, "w") as zf:
        for name, color in textures.items():
            zf.writestr(
                f"assets/minecraft/textures/block/{name}.png",
                _png_bytes(color),
            )
    return path


def _scene_texkey_of(scene):
    assert _MAGENTA_TEXKEY not in scene.texkeys
    return list(scene.texkeys)[0]


def test_model_less_stairs_uses_planks_material(tmp_path):
    pack = _texture_only_pack(
        tmp_path / "p.zip", {"oak_planks": (160, 130, 80)}
    )
    textures = load_pack(pack)
    scene = _compose(
        _single_block(
            "minecraft:oak_stairs",
            {"facing": "north", "half": "bottom", "shape": "straight"},
        ),
        0,
        textures,
    )
    assert _scene_texkey_of(scene) == "block/oak_planks"


def test_model_less_plural_material_convention(tmp_path):
    pack = _texture_only_pack(tmp_path / "p.zip", {"stone_bricks": (120, 120, 120)})
    textures = load_pack(pack)
    scene = _compose(
        _single_block(
            "minecraft:stone_brick_stairs",
            {"facing": "north", "half": "bottom", "shape": "straight"},
        ),
        0,
        textures,
    )
    assert _scene_texkey_of(scene) == "block/stone_bricks"


def test_model_less_wall_uses_material(tmp_path):
    pack = _texture_only_pack(tmp_path / "p.zip", {"cobblestone": (110, 110, 110)})
    textures = load_pack(pack)
    scene = _compose(
        _single_block(
            "minecraft:cobblestone_wall",
            {"up": "true", "north": "none", "south": "none", "east": "none",
             "west": "none"},
        ),
        0,
        textures,
    )
    assert _scene_texkey_of(scene) == "block/cobblestone"


def test_model_less_carpet_uses_wool(tmp_path):
    pack = _texture_only_pack(tmp_path / "p.zip", {"red_wool": (200, 40, 40)})
    textures = load_pack(pack)
    scene = _compose(_single_block("minecraft:red_carpet"), 0, textures)
    assert _scene_texkey_of(scene) == "block/red_wool"


def test_model_less_door_splits_top_bottom(tmp_path):
    pack = _texture_only_pack(
        tmp_path / "p.zip",
        {"oak_door_top": (200, 170, 120), "oak_door_bottom": (150, 110, 70)},
    )
    textures = load_pack(pack)
    scene = _compose(
        _single_block(
            "minecraft:oak_door",
            {"half": "lower", "facing": "east", "hinge": "left",
             "open": "false", "powered": "false"},
        ),
        0,
        textures,
    )
    keys = set(scene.texkeys)
    assert keys == {"block/oak_door_top", "block/oak_door_bottom"}
    table = list(scene.texkeys)
    upper = scene.quad_ao[
        np.isin(scene.texkey_ids, [table.index("block/oak_door_top")])
    ]
    assert len(upper) > 0


def test_direct_texture_never_overridden_by_convention(tmp_path):
    pack = _texture_only_pack(
        tmp_path / "p.zip",
        {"oak_stairs": (10, 20, 30), "oak_planks": (160, 130, 80)},
    )
    textures = load_pack(pack)
    scene = _compose(
        _single_block(
            "minecraft:oak_stairs",
            {"facing": "north", "half": "bottom", "shape": "straight"},
        ),
        0,
        textures,
    )
    assert _scene_texkey_of(scene) == "block/oak_stairs"


def test_model_less_missing_material_stays_magenta(tmp_path):
    pack = _texture_only_pack(tmp_path / "p.zip", {"stone": (125, 125, 125)})
    textures = load_pack(pack)
    scene = _compose(
        _single_block(
            "minecraft:oak_stairs",
            {"facing": "north", "half": "bottom", "shape": "straight"},
        ),
        0,
        textures,
    )
    assert _MAGENTA_TEXKEY in scene.texkeys


def test_backfill_keeps_wire_power_texkeys():
    built = builtin_quads("minecraft:redstone_wire", {"power": "8"})
    arrays = {str(k): np.zeros((16, 16, 4), np.uint8) for k in built["texkey"]}
    assert _backfill_builtin("minecraft:redstone_wire", built, arrays)
    assert built["texkey"][0] == "block/redstone_dust_dot@171,14,14"
    assert not _backfill_builtin("minecraft:redstone_wire", built, {})


def test_backfill_pane_prefers_pane_top_texture():
    built = builtin_quads("minecraft:black_stained_glass_pane", {})
    arrays = {
        "block/black_stained_glass_pane_top": np.zeros((16, 16, 4), np.uint8),
        "block/black_stained_glass": np.zeros((16, 16, 4), np.uint8),
    }
    assert _backfill_builtin("minecraft:black_stained_glass_pane", built, arrays)
    assert (built["texkey"] == "block/black_stained_glass_pane_top").all()
