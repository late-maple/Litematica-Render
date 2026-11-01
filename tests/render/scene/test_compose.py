"""Scene composition contract: quad expansion, yaw remap, culling, AO."""

import json
import zipfile

import numpy as np
import pytest
from helpers.litematica_packer import build_litematic_bytes

from litematic import loads
from litematic.render.pack import load_pack
from litematic.render.pack.resolve import (
    _MAGENTA_TEXKEY,
    Resolver,
    _backfill_builtin,
)
from litematic.render.quads import (
    QUAD_DTYPE,
    TIER_EAST,
    TIER_SOUTH,
    TIER_UP,
    unit_cube_quads,
)
from litematic.render.scene import isometric as isometric_mod
from litematic.render.scene.isometric import _compose as _compose_view
from litematic.render.scene.isometric import _vertex_ao
from litematic.render.scene.view import ISO_SLANT_DEG, ViewAxes
from litematic.render.shapes import builtin_quads


def _compose(schem, yaw, textures, entities=True):
    """Preset shim: legacy yaw index to (rotation, slant) for call sites."""
    return _compose_view(
        schem, 45.0 + 90.0 * yaw, ISO_SLANT_DEG, textures, entities
    )


_STONE = (140, 140, 140, 255)


def _textures(occl):
    """Stone cube pack whose opaque flag is the given boolean."""
    return Resolver(
        version="pack-quads",
        arrays={"block/stone": np.full((16, 16, 4), _STONE, np.uint8)},
        quads_map={"minecraft:stone[]": unit_cube_quads("block/stone")},
        occl_map={"minecraft:stone[]": occl},
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map={},
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
    assert (scene.span_w, scene.span_h) == (7, 6)
    table = list(scene.texkeys)
    assert set(table) == {"block/stone", _MAGENTA_TEXKEY}
    ids = scene.texkey_ids.tolist()
    assert ids.count(table.index("block/stone")) == 3
    assert ids.count(table.index(_MAGENTA_TEXKEY)) == 3
    assert scene.texkeys[_MAGENTA_TEXKEY].shape == (16, 16, 4)


def test_camera_sees_world_facing_tiers():
    quads = np.empty(1, QUAD_DTYPE)
    quads["vtx"] = [[(0, 16, 16), (16, 16, 16), (16, 0, 16), (0, 0, 16)]]
    quads["uv"] = [[(1, 2), (15, 2), (15, 14), (1, 14)]]
    quads["texkey"] = "block/stone"
    quads["shade"] = TIER_SOUTH
    quads["dbl"] = False
    tex = Resolver(
        version="pack-quads",
        arrays={"block/stone": np.zeros((16, 16, 4), np.uint8)},
        quads_map={"minecraft:stone[]": quads},
        occl_map={"minecraft:stone[]": False},
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map={},
    )
    schem = _single_block("minecraft:stone")
    seen = _compose_view(schem, 45.0, ISO_SLANT_DEG, tex)
    assert seen.quad_shade.tolist() == [TIER_SOUTH]
    assert seen.quad_uv.tolist() == [[[1, 2], [15, 2], [15, 14], [1, 14]]]
    assert seen.quad_vtx.tolist() == [
        [[0, 16, 16], [16, 16, 16], [16, 0, 16], [0, 0, 16]]
    ]
    hidden = _compose_view(schem, 225.0, ISO_SLANT_DEG, tex)
    assert len(hidden.x) == 0


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


def test_enclosed_cell_culled_at_every_rotation():
    cells = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]
    schem = _cell_schematic(cells)
    shell = [cell for cell in cells if cell != (1, 1, 1)]
    for yaw in range(4):
        scene = _compose(schem, yaw, _textures(True))
        seen = _scene_tuples(scene)
        assert (1, 1, 1) not in seen, yaw
        assert seen == shell, yaw


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


def test_cells_stay_world_fixed_across_rotations():
    tex = Resolver(
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
        states=None,
        parts_map={},
        mquad_map={},
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
    for yaw in range(4):
        scene = _compose(schem, yaw, tex)
        table = list(scene.texkeys)
        assert _cells_of(scene, table.index("block/stone")) == [(2, 0, 0)], yaw
        assert _cells_of(scene, table.index("block/gold_block")) == [
            (0, 0, 2)
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
    assert (scene.span_w, scene.span_h) == (25, 15)


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
    tex = Resolver(
        version="pack-quads",
        arrays={"block/stone": np.full((16, 16, 4), _STONE, np.uint8)},
        quads_map={
            "minecraft:short_grass[]": plant,
            "minecraft:stone[]": unit_cube_quads("block/stone"),
        },
        occl_map={"minecraft:short_grass[]": False, "minecraft:stone[]": True},
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map={},
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


# Neighbor face culling: quads with a cullface tier disappear when the
# neighbor occludes or pairs (same name / water family); leaves never pair.


def _cull_cube(texkey):
    quads = unit_cube_quads(texkey)
    quads["cull"] = np.arange(6, dtype=np.uint8)
    return quads


def _pair_pack(entries):
    """Full cubes with vanilla-style per-face cullfaces.

    entries: {block name: (texkey, occludes)}.
    """
    arrays, quads_map, occl_map = {}, {}, {}
    for name, (texkey, occl) in entries.items():
        arrays[texkey] = np.full((16, 16, 4), _STONE, np.uint8)
        quads_map[f"{name}[]"] = _cull_cube(texkey)
        occl_map[f"{name}[]"] = occl
    return Resolver(
        version="pack-quads",
        arrays=arrays,
        quads_map=quads_map,
        occl_map=occl_map,
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map={},
    )


def _row_scene(entries, left, right):
    """2x1x1 row of two named blocks from a _pair_pack."""
    textures = _pair_pack(entries)
    palette = [("minecraft:air", {})]
    for name in (left, right):
        if name not in [n for n, _ in palette]:
            props = {}
            if isinstance(name, tuple):
                name, props = name
            palette.append((name, props))
    indices = np.array(
        [palette.index(("minecraft:air", {})) + 1, len(palette) - 1],
        dtype=np.uint16,
    )
    def _palette_index(entry):
        name = entry if isinstance(entry, str) else entry[0]
        return next(i for i, (n, _) in enumerate(palette) if n == name)

    indices[0] = _palette_index(left)
    indices[1] = _palette_index(right)
    schem = _schematic([("Main", (0, 0, 0), (2, 1, 1), palette, indices)])
    return _compose(schem, 0, textures)


def _cell_tiers(scene):
    """(x, y, z, tier) tuples of all scene quads."""
    return set(
        zip(
            scene.x.tolist(),
            scene.y.tolist(),
            scene.z.tolist(),
            scene.quad_shade.tolist(),
        )
    )


def test_cullface_culled_by_opaque_neighbor():
    entries = {
        "minecraft:glass": ("block/glass", False),
        "minecraft:stone": ("block/stone", True),
    }
    scene = _row_scene(entries, "minecraft:glass", "minecraft:stone")
    tiers = _cell_tiers(scene)
    assert (0, 0, 0, TIER_EAST) not in tiers  # hidden by the stone
    assert (0, 0, 0, TIER_UP) in tiers  # open sky above


def test_cullface_kept_by_transparent_neighbor():
    entries = {
        "minecraft:glass": ("block/glass", False),
        "minecraft:ice": ("block/ice", False),
    }
    scene = _row_scene(entries, "minecraft:glass", "minecraft:ice")
    assert (0, 0, 0, TIER_EAST) in _cell_tiers(scene)


def test_same_name_pairs_cull():
    entries = {"minecraft:glass": ("block/glass", False)}
    scene = _row_scene(entries, "minecraft:glass", "minecraft:glass")
    assert (0, 0, 0, TIER_EAST) not in _cell_tiers(scene)


def test_different_names_do_not_pair():
    entries = {
        "minecraft:red_stained_glass": ("block/red", False),
        "minecraft:blue_stained_glass": ("block/blue", False),
    }
    scene = _row_scene(
        entries, "minecraft:red_stained_glass", "minecraft:blue_stained_glass"
    )
    assert (0, 0, 0, TIER_EAST) in _cell_tiers(scene)


def test_water_levels_pair():
    entries = {"minecraft:water": ("block/water", False)}
    scene = _row_scene(
        entries,
        ("minecraft:water", {"level": "0"}),
        ("minecraft:water", {"level": "1"}),
    )
    assert (0, 0, 0, TIER_EAST) not in _cell_tiers(scene)


def test_water_family_cross_pairs():
    entries = {
        "minecraft:water": ("block/water", False),
        "minecraft:bubble_column": ("block/bubble", False),
    }
    scene = _row_scene(
        entries, "minecraft:water", "minecraft:bubble_column"
    )
    assert (0, 0, 0, TIER_EAST) not in _cell_tiers(scene)


def test_leaves_family_never_pairs():
    entries = {
        "minecraft:oak_leaves": ("block/oak", False),
        "minecraft:birch_leaves": ("block/birch", False),
    }
    scene = _row_scene(
        entries, "minecraft:oak_leaves", "minecraft:birch_leaves"
    )
    assert (0, 0, 0, TIER_EAST) in _cell_tiers(scene)


def test_ice_pairs_same_name():
    entries = {"minecraft:ice": ("block/ice", False)}
    scene = _row_scene(entries, "minecraft:ice", "minecraft:ice")
    assert (0, 0, 0, TIER_EAST) not in _cell_tiers(scene)


def test_slime_and_honey_do_not_pair():
    entries = {
        "minecraft:slime_block": ("block/slime", False),
        "minecraft:honey_block": ("block/honey", False),
    }
    scene = _row_scene(
        entries, "minecraft:slime_block", "minecraft:honey_block"
    )
    assert (0, 0, 0, TIER_EAST) in _cell_tiers(scene)


def test_region_boundary_never_culls():
    entries = {"minecraft:ice": ("block/ice", False)}
    scene = _row_scene(entries, "minecraft:ice", "minecraft:ice")
    # The right cell's east face points out of the region: kept.
    assert (1, 0, 0, TIER_EAST) in _cell_tiers(scene)


def test_boundary_opaque_face_survives_outward_cullface():
    # An occluding block at the region edge must not cull its outward
    # face against itself (out-of-region neighbors are air, not itself).
    entries = {"minecraft:stone": ("block/stone", True)}
    schem = loads(
        build_litematic_bytes(
            [
                {
                    "name": "Main",
                    "position": (0, 0, 0),
                    "size": (1, 1, 1),
                    "palette": [("minecraft:air", {}), ("minecraft:stone", {})],
                    "indices_yzx": np.array([1], dtype=np.uint16),
                }
            ]
        )
    )
    scene = _compose(schem, 0, _pair_pack(entries))
    tiers = _cell_tiers(scene)
    assert (0, 0, 0, TIER_EAST) in tiers
    assert (0, 0, 0, TIER_UP) in tiers
    assert (0, 0, 0, TIER_SOUTH) in tiers


# AO lookups go through a False-padded flat grid with int32 index math;
# the padded form must equal the classic masked 3D lookup everywhere.


def _reference_ao(quads, cells, grid):
    """The classic masked 3D-grid AO formula, kept as the oracle."""
    from litematic.render.quads import AO_LEVELS
    from litematic.render.scene.isometric import _TIER_AXES, _TIER_NORMALS

    def lookup(points):
        sx, sy, sz = grid.shape
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        valid = (
            (x >= 0) & (x < sx) & (y >= 0) & (y < sy) & (z >= 0) & (z < sz)
        )
        out = np.zeros(points.shape[:-1], bool)
        if valid.any():
            out[valid] = grid[x[valid], y[valid], z[valid]]
        return out

    tier = quads["shade"].astype(np.int64)
    axes = _TIER_AXES[tier]
    normals = _TIER_NORMALS[tier]
    e1 = np.zeros((len(tier), 3), np.int64)
    e1[np.arange(len(tier)), axes[:, 1]] = 1
    e2 = np.zeros((len(tier), 3), np.int64)
    e2[np.arange(len(tier)), axes[:, 2]] = 1
    vtx = quads["vtx"]
    pick1 = axes[:, 1][:, None, None]
    pick2 = axes[:, 2][:, None, None]
    s1 = np.sign(np.take_along_axis(vtx, pick1, axis=2)[:, :, 0] - 8.0)
    s2 = np.sign(np.take_along_axis(vtx, pick2, axis=2)[:, :, 0] - 8.0)
    s1 = s1.astype(np.int64)
    s2 = s2.astype(np.int64)
    base = cells[:, None, :] + normals[:, None, :]
    d1 = s1[:, :, None] * e1[:, None, :]
    d2 = s2[:, :, None] * e2[:, None, :]
    side1 = lookup(base + d1).astype(np.int64)
    side2 = lookup(base + d2).astype(np.int64)
    corner = lookup(base + d1 + d2).astype(np.int64)
    level = np.where(side1 & side2, 0, 3 - (side1 + side2 + corner))
    ao = np.asarray(AO_LEVELS, np.float32)[level]
    ao[quads["dbl"]] = 1.0
    return ao


def _pad_flat(grid):
    padded = np.pad(grid, 1)
    strides = ((grid.shape[1] + 2) * (grid.shape[2] + 2), grid.shape[2] + 2, 1)
    return padded.ravel(), strides


def test_padded_ao_equals_reference():
    rng = np.random.default_rng(7)
    grid = rng.random((5, 4, 6)) < 0.4
    flat, strides = _pad_flat(grid)
    quads = np.empty(24, QUAD_DTYPE)
    quads["shade"] = np.tile(np.arange(6, dtype=np.uint8), 4)
    quads["vtx"] = np.tile(
        np.array([[0, 0, 0], [0, 8, 0], [0, 8, 16], [0, 0, 16]], np.float32),
        (24, 1, 1),
    )
    quads["dbl"] = np.zeros(24, bool)
    quads["dbl"][3] = True  # a double-sided row short-circuits to 1.0
    cells = np.stack(
        [rng.integers(0, 5, 24), rng.integers(0, 4, 24), rng.integers(0, 6, 24)],
        axis=1,
    )
    cells[0] = (0, 0, 0)
    cells[1] = (4, 3, 5)
    out = _vertex_ao(
        quads["vtx"], quads["shade"], quads["dbl"], cells, flat, strides
    )
    assert out.dtype == np.float32
    assert (out == _reference_ao(quads, cells, grid)).all()


def test_ao_boundary_cells_match_reference():
    grid = np.zeros((3, 3, 3), bool)
    grid[0, :, :] = True
    grid[:, 2, 1] = True
    flat, strides = _pad_flat(grid)
    quads = np.empty(8, QUAD_DTYPE)
    quads["shade"] = np.array([1, 2, 3, 4, 5, 0, 1, 2], np.uint8)
    quads["vtx"] = np.tile(
        np.array([[0, 0, 0], [0, 8, 0], [0, 8, 16], [0, 0, 16]], np.float32),
        (8, 1, 1),
    )
    quads["dbl"] = np.zeros(8, bool)
    cells = np.array(
        [(0, 0, 0), (2, 2, 2), (0, 2, 1), (1, 0, 2),
         (2, 1, 0), (0, 1, 1), (1, 2, 0), (2, 0, 1)], np.int64
    )
    out = _vertex_ao(
        quads["vtx"], quads["shade"], quads["dbl"], cells, flat, strides
    )
    assert (out == _reference_ao(quads, cells, grid)).all()


# Streaming compose: slab height must not change the scene; boundaries
# across slabs behave like in-region neighbors; the arena trims slack.


def _box_schematic(size, name):
    sx, sy, sz = size
    palette = [("minecraft:air", {}), (name, {})]
    indices = np.ones(sx * sy * sz, dtype=np.uint16)
    return _schematic([("Main", (0, 0, 0), size, palette, indices)])


def _scene_rows(scene):
    """Scene quads as a sortable row multiset (order-independent compare)."""
    return sorted(
        zip(
            scene.x.tolist(),
            scene.y.tolist(),
            scene.z.tolist(),
            scene.quad_shade.tolist(),
            [r.tobytes() for r in scene.quad_vtx],
            [r.tobytes() for r in scene.quad_uv],
            [r.tobytes() for r in scene.quad_ao],
            scene.quad_dbl.tolist(),
            scene.texkey_ids.tolist(),
        )
    )


def test_slab_height_invariance(monkeypatch):
    # Slab height changes quad ORDER (x-major per slab), never content.
    schem = _box_schematic((3, 130, 3), "minecraft:stone")
    textures = _pair_pack({"minecraft:stone": ("block/stone", True)})
    rows = None
    for slab in (130, 64, 7):
        monkeypatch.setattr(isometric_mod, "_Y_SLAB", slab)
        scene = _compose(schem, 0, textures)
        if rows is None:
            rows = _scene_rows(scene)
        assert _scene_rows(scene) == rows


def test_slab_boundary_ao_and_cull():
    palette = [
        ("minecraft:air", {}),
        ("minecraft:glass", {}),
        ("minecraft:stone", {}),
    ]
    indices = np.zeros(3 * 128 * 3, dtype=np.uint16).reshape(128, 3, 3)
    indices[63, 1, 1] = 1  # glass right under the slab edge
    indices[64, :, :] = 2  # stone layer right above, across y=64
    indices = indices.ravel()
    schem = _schematic([("Main", (0, 0, 0), (3, 128, 3), palette, indices)])
    textures = _pair_pack(
        {
            "minecraft:glass": ("block/glass", False),
            "minecraft:stone": ("block/stone", True),
        }
    )
    scene = _compose(schem, 0, textures)
    tiers = _cell_tiers(scene)
    glass_y = int(scene.y.min())  # the glass cell after min-shift
    assert (0, glass_y, 0, TIER_UP) not in tiers  # culled by the stone

    # Without a cullface the face survives and AO must see the stone.
    palette[1] = ("minecraft:plain", {})
    plain = loads(
        build_litematic_bytes(
            [
                {
                    "name": "Main",
                    "position": (0, 0, 0),
                    "size": (3, 128, 3),
                    "palette": palette,
                    "indices_yzx": indices,
                }
            ]
        )
    )
    plain_textures = Resolver(
        version="pack-quads",
        arrays={
            "block/plain": np.full((16, 16, 4), _STONE, np.uint8),
            "block/stone2": np.full((16, 16, 4), _STONE, np.uint8),
        },
        quads_map={
            "minecraft:plain[]": unit_cube_quads("block/plain"),
            "minecraft:stone[]": _cull_cube("block/stone2"),
        },
        occl_map={"minecraft:plain[]": False, "minecraft:stone[]": True},
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map={},
    )
    scene = _compose(plain, 0, plain_textures)
    plain_y = int(scene.y.min())
    rows = (scene.y == plain_y) & (scene.quad_shade == TIER_UP)
    assert rows.any() and float(scene.quad_ao[rows].max()) < 1.0


def test_arena_trim_removes_cull_slack():
    entries = {"minecraft:stone": ("block/stone", True)}
    scene = _row_scene(entries, "minecraft:stone", "minecraft:stone")
    # Capacity is 2 cells x 3 camera faces; one face is culled.
    assert len(scene.x) == 5


def test_scene_fields_do_not_pin_arena():
    entries = {"minecraft:stone": ("block/stone", True)}
    scene = _row_scene(entries, "minecraft:stone", "minecraft:stone")
    for arr in (
        scene.x,
        scene.y,
        scene.z,
        scene.quad_vtx,
        scene.quad_uv,
        scene.quad_ao,
        scene.quad_shade,
        scene.quad_dbl,
        scene.texkey_ids,
    ):
        assert arr.base is None


def test_degenerate_regions():
    textures = _pair_pack({"minecraft:stone": ("block/stone", True)})
    assert len(_compose(_single_block("minecraft:air"), 0, textures).x) == 0
    empty = _box_schematic((2, 1, 2), "minecraft:air")
    assert len(_compose(empty, 0, textures).x) == 0


# Entity mounting: entities compile to float-offset quad rows appended to
# the block pass, never occluding, skipped when outside the block volume.


def _entity_textures():
    return Resolver(
        version="pack-quads",
        arrays={
            "block/stone": np.full((16, 16, 4), _STONE, np.uint8),
            "item/wheat_seeds": np.full((16, 16, 4), (160, 120, 40, 255), np.uint8),
            "block/item_frame": np.full((16, 16, 4), (90, 60, 30, 255), np.uint8),
            "block/birch_planks": np.full((16, 16, 4), (196, 179, 140, 255), np.uint8),
            "entity/armorstand/armorstand": np.full(
                (64, 64, 4), (120, 100, 90, 255), np.uint8
            ),
        },
        quads_map={"minecraft:stone[]": unit_cube_quads("block/stone")},
        occl_map={"minecraft:stone[]": True},
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map={"__frame_pad": _frame_pad_quads()},
        item_display={"minecraft:wheat_seeds": {"flat": "item/wheat_seeds"}},
    )


def _frame_pad_quads():
    pad = np.zeros(2, QUAD_DTYPE)
    pad["texkey"][:] = "block/item_frame"
    pad["vtx"][:, :, 2] = 15.5
    return pad


def _entity_schematic(entities, size=(2, 2, 2)):
    palette = [("minecraft:air", {}), ("minecraft:stone", {})]
    n = size[0] * size[1] * size[2]
    indices = np.ones(n, dtype=np.uint16)
    return loads(
        build_litematic_bytes(
            [
                {
                    "name": "Main",
                    "position": (0, 0, 0),
                    "size": size,
                    "palette": palette,
                    "indices_yzx": indices,
                    "entities": entities,
                }
            ]
        )
    )


def _item_nbt(x, y, z, item="minecraft:wheat_seeds"):
    return {
        "id": "minecraft:item",
        "Pos": [float(x), float(y), float(z)],
        "Item": {"id": item, "Count": 1},
    }


def _entity_rows_of(scene, key_fragment):
    ids = [i for i, k in enumerate(scene.texkeys) if key_fragment in k]
    if not ids:
        return np.zeros(len(scene.texkey_ids), bool)
    return np.isin(scene.texkey_ids, ids)


def test_item_entity_mounts_with_fractional_offsets():
    schem = _entity_schematic(
        [_item_nbt(0.5, 0.25, 0.5)],
    )
    scene = _compose(schem, 0, _entity_textures())
    rows = _entity_rows_of(scene, "item/wheat_seeds")
    assert rows.any()
    vtx = scene.quad_vtx[rows]
    x = vtx[..., 0]
    # x fraction 0.5 = cell-centered: builders already sit at the center
    assert x.min() == pytest.approx(4.0)
    y = vtx[..., 1]
    assert y.min() == pytest.approx(2.6 + 4.0)  # y fraction 0.25 * 16


def test_entities_outside_region_volume_skipped():
    schem = _entity_schematic([_item_nbt(5.0, 0.0, 0.0)])
    scene = _compose(schem, 0, _entity_textures())
    assert not _entity_rows_of(scene, "item/wheat_seeds").any()


def test_entities_false_mounts_nothing():
    schem = _entity_schematic([_item_nbt(0.5, 0.0, 0.5)])
    full = _compose(schem, 0, _entity_textures())
    bare = _compose(schem, 0, _entity_textures(), entities=False)
    assert _entity_rows_of(full, "item/wheat_seeds").any()
    assert not _entity_rows_of(bare, "item/wheat_seeds").any()


def test_frame_and_stand_mount_with_real_texkeys():
    entities = [
        {
            "id": "minecraft:item_frame",
            "Pos": [0.5, 1.0, 0.5],
            "Facing": 3,
            "Item": {"id": "minecraft:wheat_seeds", "Count": 1},
        },
        {"id": "minecraft:armor_stand", "Pos": [1.5, 0.0, 1.5]},
    ]
    scene = _compose(_entity_schematic(entities), 0, _entity_textures())
    assert _entity_rows_of(scene, "block/item_frame").any()
    assert _entity_rows_of(scene, "entity/armorstand").any()


def test_unknown_entity_types_skip_silently():
    entities = [{"id": "minecraft:pig", "Pos": [0.5, 0.0, 0.5]}]
    scene = _compose(_entity_schematic(entities), 0, _entity_textures())
    assert len(scene.x) == sum(
        1 for _ in scene.x
    ) - _entity_rows_of(scene, "item").sum()


def test_entity_anchors_stay_world_fixed():
    schem = _entity_schematic([_item_nbt(0.5, 0.0, 0.5)])
    for yaw in range(4):
        scene = _compose(schem, yaw, _entity_textures())
        rows = _entity_rows_of(scene, "item/wheat_seeds")
        assert rows.any(), yaw
        anchor_x = scene.x[rows][0]
        anchor_z = scene.z[rows][0]
        assert (anchor_x, anchor_z) == (0, 0), yaw


def test_span_survives_axis_ulp_noise(monkeypatch):
    """Platform trig differs by 1 ULP; the span ceil must not flip."""
    real = isometric_mod.view_axes

    def noisy(rotation, slant):
        axes = real(rotation, slant)
        return ViewAxes(
            np.nextafter(axes.right, np.inf),
            np.nextafter(axes.up, -np.inf),
            np.nextafter(axes.toward, np.inf),
            axes.tiers,
        )

    monkeypatch.setattr(isometric_mod, "view_axes", noisy)
    palette = [
        ("minecraft:air", {}),
        ("minecraft:stone", {}),
        ("minecraft:gold_block", {}),
    ]
    indices = np.array([1, 2], dtype=np.uint16)
    schem = _schematic([("Main", (0, 0, 0), (2, 1, 1), palette, indices)])
    scene = _compose(schem, 0, _textures(True))
    assert (scene.span_w, scene.span_h) == (7, 6)
