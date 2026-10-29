"""Pack extraction contract: blockstates→models→textures→npz (state-keyed)."""

import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from litematic.render.errors import RenderError
from litematic.render.pack import load_pack
from litematic.render.pack.elements import _FACE_GEOM, _compile_elements
from litematic.render.pack.extract import (
    ExtractSummary,
    _CacheWriteError,
    _extract_pack,
)
from litematic.render.pack.texture import _resolve_ref
from litematic.render.quads import (
    CULL_NONE,
    QUAD_DTYPE,
    TIER_DOWN,
    TIER_EAST,
    TIER_NORTH,
    TIER_SOUTH,
    TIER_UP,
    TIER_WEST,
    sort_quads,
    unit_cube_quads,
)


def _solid_png(color, size=(16, 16)):
    image = Image.new("RGBA", size, color)
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def _strip_png(top, bottom):
    image = Image.new("RGBA", (16, 32))
    for y in range(32):
        for x in range(16):
            image.putpixel((x, y), top if y < 16 else bottom)
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def _quadrant_png():
    image = Image.new("RGBA", (32, 32))
    for y in range(32):
        for x in range(32):
            color = (255, 0, 0, 255) if x < 16 and y < 16 else (0, 255, 0, 255)
            image.putpixel((x, y), color)
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def _garbage_png():
    return b"\x89PNG\r\n\x1a\nnot really a png"


def _blockstates_variants(variants):
    return json.dumps({"variants": variants})


def _blockstates(model, x=None, y=None):
    variant = {"model": model}
    if x is not None:
        variant["x"] = x
    if y is not None:
        variant["y"] = y
    return _blockstates_variants({"": variant})


def _model(faces, textures=None, parent=None, to=(16, 16, 16)):
    model = {}
    if parent is not None:
        model["parent"] = parent
    if textures is not None:
        model["textures"] = textures
    model["elements"] = [{"from": [0, 0, 0], "to": list(to), "faces": faces}]
    return json.dumps(model)


def _two_box_model(faces):
    """Build a model with two stacked unit-cube box elements."""
    return json.dumps(
        {
            "elements": [
                {"from": [0, 0, 0], "to": [16, 16, 16], "faces": faces},
                {"from": [0, 0, 0], "to": [16, 16, 16], "faces": faces},
            ]
        }
    )


def _all_faces(ref="#all"):
    face = {"texture": ref}
    names = ("down", "up", "north", "south", "west", "east")
    return {name: dict(face) for name in names}


def _make_pack(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _extract(tmp_path, entries):
    dest = tmp_path / "out.npz"
    _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    return dest


def _stone_entries(**overrides):
    entries = {
        "assets/minecraft/blockstates/stone.json": _blockstates("block/stone"),
        "assets/minecraft/models/block/stone.json": _model(_all_faces("block/stone")),
        "assets/minecraft/textures/block/stone.png": _solid_png((200, 30, 30, 255)),
    }
    entries.update(overrides)
    return entries


def _load_npz(dest):
    return np.load(dest, allow_pickle=False)


def _quad_ref(data, block, state, tier):
    """Return the texkey of the block state's quad on one face tier."""
    quads = data[f"quads/{block}[{state}]"]
    return str(next(q["texkey"] for q in quads if int(q["shade"]) == tier))


def _tex(data, key):
    return data[f"tex/{key}"]


def test_plain_block_stateless_entry(tmp_path):
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", _stone_entries()), dest)
    assert isinstance(summary, ExtractSummary)
    assert summary.blocks == 1
    assert summary.skipped == 0
    data = _load_npz(dest)
    assert data["meta/version"].item() == "pack-quads.5"
    quads = data["quads/minecraft:stone[]"]
    assert quads.dtype == QUAD_DTYPE
    assert quads["texkey"].tolist() == ["block/stone"] * 6
    tex = _tex(data, "block/stone")
    assert tex.shape == (16, 16, 4)
    assert tex.dtype == np.uint8
    assert (tex == (200, 30, 30, 255)).all()
    assert int(data["occl/minecraft:stone[]"].item()) == 1


def test_hash_ref_resolves_through_model_textures(tmp_path):
    entries = {
        "assets/minecraft/blockstates/pillar.json": _blockstates("block/pillar"),
        "assets/minecraft/models/block/pillar.json": _model(
            {
                **_all_faces("#side"),
                "up": {"texture": "#end"},
                "down": {"texture": "#end"},
            },
            textures={"side": "block/side_tex", "end": "block/end_tex"},
        ),
        "assets/minecraft/textures/block/side_tex.png": _solid_png((1, 2, 3, 255)),
        "assets/minecraft/textures/block/end_tex.png": _solid_png((4, 5, 6, 255)),
    }
    data = _load_npz(_extract(tmp_path, entries))
    quads = data["quads/minecraft:pillar[]"]
    by_tier = {int(q["shade"]): str(q["texkey"]) for q in quads}
    assert by_tier[TIER_UP] == "block/end_tex"
    assert by_tier[TIER_EAST] == "block/side_tex"
    assert (_tex(data, "block/end_tex") == (4, 5, 6, 255)).all()


def test_parent_chain_merge_child_overrides(tmp_path):
    entries = {
        "assets/minecraft/blockstates/child.json": _blockstates("block/child"),
        "assets/minecraft/models/block/parent.json": _model(
            _all_faces("#all"), textures={"all": "block/old"}
        ),
        "assets/minecraft/models/block/child.json": json.dumps(
            {"parent": "block/parent", "textures": {"all": "block/new"}}
        ),
        "assets/minecraft/textures/block/old.png": _solid_png((9, 9, 9, 255)),
        "assets/minecraft/textures/block/new.png": _solid_png((7, 7, 7, 255)),
    }
    data = _load_npz(_extract(tmp_path, entries))
    quads = data["quads/minecraft:child[]"]
    assert {str(k) for k in quads["texkey"]} == {"block/new"}
    assert (_tex(data, "block/new") == (7, 7, 7, 255)).all()


def test_variant_y_rotation_swaps_faces(tmp_path):
    channel = {"up": 1, "north": 2, "east": 3, "south": 4, "west": 5, "down": 6}
    faces = {name: {"texture": f"block/{name}_tex"} for name in channel}
    entries = {
        "assets/minecraft/blockstates/front.json": _blockstates("block/front", y=90),
        "assets/minecraft/models/block/front.json": _model(faces),
    }
    for name, value in channel.items():
        entries[f"assets/minecraft/textures/block/{name}_tex.png"] = _solid_png(
            (10, 20, value, 255)
        )
    data = _load_npz(_extract(tmp_path, entries))
    quads = data["quads/minecraft:front[]"]
    by_tier = {int(q["shade"]): str(q["texkey"]) for q in quads}
    assert by_tier[TIER_EAST] == "block/north_tex"
    assert by_tier[TIER_SOUTH] == "block/east_tex"
    assert by_tier[TIER_UP] == "block/up_tex"


def test_all_variants_extracted_per_state(tmp_path):
    faces = {
        "north": {"texture": "block/front_tex"},
        **{
            name: {"texture": "block/side_tex"}
            for name in ("up", "east", "south", "west", "down")
        },
    }
    entries = {
        "assets/minecraft/blockstates/furnace.json": _blockstates_variants(
            {
                "facing=north": {"model": "block/furnace"},
                "facing=south": {"model": "block/furnace", "y": 180},
            }
        ),
        "assets/minecraft/models/block/furnace.json": _model(faces),
        "assets/minecraft/textures/block/front_tex.png": _solid_png((9, 9, 9, 255)),
        "assets/minecraft/textures/block/side_tex.png": _solid_png((1, 1, 1, 255)),
    }
    data = _load_npz(_extract(tmp_path, entries))
    south = data["quads/minecraft:furnace[facing=south]"]
    by_tier = {int(q["shade"]): str(q["texkey"]) for q in south}
    assert by_tier[TIER_SOUTH] == "block/front_tex"
    north = data["quads/minecraft:furnace[facing=north]"]
    by_tier = {int(q["shade"]): str(q["texkey"]) for q in north}
    assert by_tier[TIER_NORTH] == "block/front_tex"
    assert by_tier[TIER_EAST] == "block/side_tex"


def test_state_key_pairs_sorted_by_property_name(tmp_path):
    entries = _stone_entries()
    entries["assets/minecraft/blockstates/unsorted.json"] = _blockstates_variants(
        {"shape=straight,half=bottom,facing=east": {"model": "block/stone"}}
    )
    data = _load_npz(_extract(tmp_path, entries))
    key = "quads/minecraft:unsorted[facing=east,half=bottom,shape=straight]"
    assert key in data.files


def test_tintindex_applies_constant(tmp_path):
    faces = _all_faces("block/grass")
    faces["up"]["tintindex"] = 0
    entries = {
        "assets/minecraft/blockstates/grass_block.json": _blockstates(
            "block/grass_block"
        ),
        "assets/minecraft/models/block/grass_block.json": _model(faces),
        "assets/minecraft/textures/block/grass.png": _solid_png((255, 255, 255, 255)),
    }
    data = _load_npz(_extract(tmp_path, entries))
    quads = data["quads/minecraft:grass_block[]"]
    by_tier = {int(q["shade"]): str(q["texkey"]) for q in quads}
    assert by_tier[TIER_UP] == "block/grass@145,189,89"
    assert (_tex(data, "block/grass@145,189,89") == (145, 189, 89, 255)).all()
    assert by_tier[TIER_SOUTH] == "block/grass"
    assert (_tex(data, "block/grass") == (255, 255, 255, 255)).all()


def test_redstone_power_tint_interpolates(tmp_path):
    faces = {name: {"texture": "block/dust", "tintindex": 0} for name in _all_faces()}
    entries = {
        "assets/minecraft/blockstates/redstone_wire.json": _blockstates_variants(
            {"power=0": {"model": "block/dust"}, "power=15": {"model": "block/dust"}}
        ),
        "assets/minecraft/models/block/dust.json": _model(faces),
        "assets/minecraft/textures/block/dust.png": _solid_png((255, 255, 255, 255)),
    }
    data = _load_npz(_extract(tmp_path, entries))
    low = _tex(
        data, _quad_ref(data, "minecraft:redstone_wire", "power=0", TIER_UP)
    )
    high = _tex(
        data, _quad_ref(data, "minecraft:redstone_wire", "power=15", TIER_UP)
    )
    assert (low[0, 0] == (76, 0, 0, 255)).all()
    assert (high[0, 0] == (255, 26, 26, 255)).all()


@pytest.mark.parametrize(
    ("block", "color"),
    [
        ("bubble_column", (63, 118, 228)),
        ("vine", (119, 171, 47)),
        ("water_cauldron", (255, 255, 255)),
    ],
)
def test_tint_matching_rules(tmp_path, block, color):
    entries = {
        f"assets/minecraft/blockstates/{block}.json": _blockstates(f"block/{block}"),
        f"assets/minecraft/models/block/{block}.json": _model(
            {
                **_all_faces(f"block/{block}"),
                "up": {"texture": f"block/{block}", "tintindex": 0},
            }
        ),
        f"assets/minecraft/textures/block/{block}.png": _solid_png(
            (255, 255, 255, 255)
        ),
    }
    data = _load_npz(_extract(tmp_path, entries))
    tex = _tex(data, _quad_ref(data, f"minecraft:{block}", "", TIER_UP))
    assert (tex[0, 0] == (*color, 255)).all()


def test_animation_strip_takes_first_frame(tmp_path):
    entries = _stone_entries()
    entries["assets/minecraft/textures/block/stone.png"] = _strip_png(
        (200, 0, 0, 255), (0, 0, 200, 255)
    )
    entries["assets/minecraft/textures/block/stone.png.mcmeta"] = (
        '{"animation":{"frametime":2}}'
    )
    data = _load_npz(_extract(tmp_path, entries))
    assert (_tex(data, "block/stone") == (200, 0, 0, 255)).all()


def test_hi_res_quadrant_kept_native(tmp_path):
    entries = _stone_entries()
    entries["assets/minecraft/textures/block/stone.png"] = _quadrant_png()
    data = _load_npz(_extract(tmp_path, entries))
    tex = _tex(data, "block/stone")
    assert tex.shape == (32, 32, 4)
    assert (tex[0, 0][:3] == (255, 0, 0)).all()
    assert (tex[31, 31][:3] == (0, 255, 0)).all()


def test_partial_face_model_emits_its_quads(tmp_path):
    entries = _stone_entries()
    entries["assets/minecraft/blockstates/torch.json"] = _blockstates("block/torch")
    entries["assets/minecraft/models/block/torch.json"] = _model(
        {"up": {"texture": "block/stone"}}
    )
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    assert summary.blocks == 2
    assert summary.skipped == 0
    data = _load_npz(dest)
    quads = data["quads/minecraft:torch[]"]
    assert len(quads) == 1
    assert int(quads[0]["shade"]) == TIER_UP
    assert int(data["occl/minecraft:torch[]"].item()) == 0


def test_malformed_model_skips_block(tmp_path):
    entries = _stone_entries()
    entries["assets/minecraft/blockstates/bad.json"] = _blockstates("block/bad")
    entries["assets/minecraft/models/block/bad.json"] = "{not json"
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    assert summary.blocks == 1
    assert summary.skipped == 1
    data = _load_npz(dest)
    assert "quads/minecraft:stone[]" in data.files
    assert not [k for k in data.files if k.startswith("quads/minecraft:bad[")]


def test_broken_png_skips_block(tmp_path):
    entries = _stone_entries()
    entries["assets/minecraft/textures/block/stone.png"] = _garbage_png()
    summary = _extract_pack(
        _make_pack(tmp_path / "pack.zip", entries), tmp_path / "out.npz"
    )
    assert summary.blocks == 0
    assert summary.skipped == 1


def test_hash_ref_chain_follows_to_path(tmp_path):
    entries = {
        "assets/minecraft/blockstates/chained.json": _blockstates("block/chained"),
        "assets/minecraft/models/block/chained.json": _model(
            _all_faces("#outer"),
            textures={"outer": "#inner", "inner": "block/deep"},
        ),
        "assets/minecraft/textures/block/deep.png": _solid_png((11, 22, 33, 255)),
    }
    data = _load_npz(_extract(tmp_path, entries))
    assert _quad_ref(data, "minecraft:chained", "", TIER_UP) == "block/deep"
    assert (_tex(data, "block/deep") == (11, 22, 33, 255)).all()


def test_ref_with_namespace_prefix_resolves(tmp_path):
    entries = {
        "assets/minecraft/blockstates/namespaced.json": _blockstates(
            "block/namespaced"
        ),
        "assets/minecraft/models/block/namespaced.json": _model(
            _all_faces("minecraft:block/prefixed")
        ),
        "assets/minecraft/textures/block/prefixed.png": _solid_png((44, 55, 66, 255)),
    }
    data = _load_npz(_extract(tmp_path, entries))
    assert _quad_ref(data, "minecraft:namespaced", "", TIER_UP) == "block/prefixed"
    assert (_tex(data, "block/prefixed") == (44, 55, 66, 255)).all()


def test_traversal_ref_skips_block(tmp_path):
    entries = {
        "assets/minecraft/blockstates/evil.json": _blockstates("block/evil"),
        "assets/minecraft/models/block/evil.json": _model(_all_faces("../../secret")),
    }
    summary = _extract_pack(
        _make_pack(tmp_path / "pack.zip", entries), tmp_path / "out.npz"
    )
    assert summary.blocks == 0
    assert summary.skipped == 1


@pytest.mark.parametrize(
    "pack_bytes", [b"plain text file", b"", b"\x00\x01\x02garbagezip"]
)
def test_rejects_non_zip_source(tmp_path, pack_bytes):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(pack_bytes)
    with pytest.raises(RenderError, match="extract:"):
        _extract_pack(bad, tmp_path / "out.npz")


def test_rejects_zip_without_blockstates(tmp_path):
    empty = _make_pack(tmp_path / "empty.zip", {"pack.mcmeta": "{}"})
    with pytest.raises(RenderError, match="blockstates"):
        _extract_pack(empty, tmp_path / "out.npz")


def test_rejects_missing_source(tmp_path):
    with pytest.raises(RenderError, match="extract:"):
        _extract_pack(tmp_path / "nope.zip", tmp_path / "out.npz")


def test_npz_bytes_deterministic(tmp_path):
    pack = _make_pack(tmp_path / "pack.zip", _stone_entries())
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    _extract_pack(pack, first)
    _extract_pack(pack, second)
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as data:
        quads = data["quads/minecraft:stone[]"]
    assert np.array_equal(quads, sort_quads(quads))


def test_missing_pillow_raises_render_error(tmp_path, monkeypatch):
    import sys

    pack = _make_pack(tmp_path / "pack.zip", _stone_entries())
    dest = tmp_path / "out.npz"
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(RenderError, match="Pillow is required"):
        _extract_pack(pack, dest)
    assert not dest.exists()


def test_cross_namespace_bare_refs_do_not_collide(tmp_path):
    entries = {}
    for ns, color in (("alpha", (10, 10, 10)), ("beta", (20, 20, 20))):
        entries[f"assets/{ns}/blockstates/shared.json"] = _blockstates(
            "block/plain"
        )
        entries[f"assets/{ns}/models/block/plain.json"] = _model(
            _all_faces("block/shared")
        )
        entries[f"assets/{ns}/textures/block/shared.png"] = _solid_png(
            (*color, 255)
        )
    data = _load_npz(_extract(tmp_path, entries))
    alpha_tex = _tex(data, "alpha:block/shared")
    beta_tex = _tex(data, "beta:block/shared")
    assert (alpha_tex == (10, 10, 10, 255)).all()
    assert (beta_tex == (20, 20, 20, 255)).all()
    assert (
        _tex(data, _quad_ref(data, "alpha:shared", "", TIER_UP)) == (10, 10, 10, 255)
    ).all()
    assert (
        _tex(data, _quad_ref(data, "beta:shared", "", TIER_UP)) == (20, 20, 20, 255)
    ).all()


def test_unwritable_dest_raises_render_error(tmp_path):
    pack = _make_pack(tmp_path / "pack.zip", _stone_entries())
    dest = tmp_path / "missing_dir" / "out.npz"
    with pytest.raises(RenderError, match="extract:"):
        _extract_pack(pack, dest)


def _texture_only_entries():
    return {
        "assets/minecraft/textures/block/stone.png": _solid_png(
            (200, 30, 30, 255)
        ),
        "assets/minecraft/textures/block/grass_block.png": _solid_png(
            (255, 255, 255, 255)
        ),
        "assets/modmenu/textures/gui/mods_button.png": _solid_png(
            (9, 9, 9, 255)
        ),
        "assets/minecraft/textures/block/nested/deep.png": _solid_png(
            (1, 2, 3, 255)
        ),
        "assets/minecraft/textures/item/stick.png": _solid_png((4, 5, 6, 255)),
        "assets/minecraft/textures/block/UPPER.png": _solid_png((7, 7, 7, 255)),
    }


def test_texture_only_pack_falls_back_to_block_textures(tmp_path):
    data = _load_npz(_extract(tmp_path, _texture_only_entries()))
    quads = data["quads/minecraft:stone[]"]
    assert len(quads) == 6
    assert set(quads["texkey"].tolist()) == {"block/stone"}
    assert (_tex(data, "block/stone") == (200, 30, 30, 255)).all()
    assert not [k for k in data.files if k.startswith("quads/modmenu:")]
    assert not [k for k in data.files if k.startswith("quads/minecraft:deep[")]
    assert not [k for k in data.files if k.startswith("quads/minecraft:stick[")]
    assert not [k for k in data.files if k.startswith("quads/minecraft:UPPER[")]


def test_texture_fallback_tints_by_block_name(tmp_path):
    data = _load_npz(_extract(tmp_path, _texture_only_entries()))
    quads = data["quads/minecraft:grass_block[]"]
    key = str(quads["texkey"][0])
    assert key == "block/grass_block@145,189,89"
    assert (_tex(data, key) == (145, 189, 89, 255)).all()


def test_fallback_skips_blockstate_covered_blocks(tmp_path):
    entries = _stone_entries(
        **{
            "assets/minecraft/textures/block/dirt.png": _solid_png(
                (120, 85, 60, 255)
            )
        }
    )
    dest = tmp_path / "out.npz"
    summary = _extract_pack(
        _make_pack(tmp_path / "pack.zip", entries), dest
    )
    data = _load_npz(dest)
    assert "quads/minecraft:dirt[]" in data.files
    assert summary.blocks == 2
    assert summary.skipped == 0


def test_foreign_namespace_texture_fallback(tmp_path):
    entries = {
        "assets/create/textures/block/limestone.png": _solid_png(
            (180, 180, 170, 255)
        )
    }
    data = _load_npz(_extract(tmp_path, entries))
    quads = data["quads/create:limestone[]"]
    assert set(quads["texkey"].tolist()) == {"create:block/limestone"}


def test_pack_without_blockstates_or_block_textures_rejected(tmp_path):
    entries = {"assets/minecraft/textures/item/stick.png": _solid_png((4, 5, 6, 255))}
    with pytest.raises(RenderError, match="not a resource pack"):
        _extract(tmp_path, entries)


def _stub_water_entries():
    return {
        "assets/minecraft/blockstates/water.json": _blockstates("block/water"),
        "assets/minecraft/models/block/water.json": json.dumps(
            {"textures": {"particle": "block/water_still"}}
        ),
        "assets/minecraft/textures/block/water_still.png": _solid_png(
            (255, 255, 255, 255)
        ),
    }


def test_stub_model_without_texture_stays_unresolved(tmp_path):
    entries = {
        "assets/minecraft/blockstates/portal.json": _blockstates("block/portal"),
        "assets/minecraft/models/block/portal.json": json.dumps(
            {"textures": {"particle": "block/missing"}}
        ),
    }
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    data = _load_npz(dest)
    assert summary.skipped == 1
    assert not [k for k in data.files if k.startswith("quads/minecraft:portal[")]


def test_stub_parent_chain_supplies_texture(tmp_path):
    entries = {
        "assets/minecraft/blockstates/flow.json": _blockstates("block/flow"),
        "assets/minecraft/models/block/flow.json": json.dumps(
            {"parent": "block/base"}
        ),
        "assets/minecraft/models/block/base.json": json.dumps(
            {"textures": {"particle": "block/still"}}
        ),
        "assets/minecraft/textures/block/still.png": _solid_png(
            (10, 20, 30, 255)
        ),
    }
    data = _load_npz(_extract(tmp_path, entries))
    assert _quad_ref(data, "minecraft:flow", "", TIER_UP) == "block/still"


def test_builtin_parent_stays_unresolved(tmp_path):
    entries = {
        "assets/minecraft/blockstates/chest.json": _blockstates("block/chest"),
        "assets/minecraft/models/block/chest.json": json.dumps(
            {"parent": "builtin/entity", "textures": {"particle": "block/planks"}}
        ),
        "assets/minecraft/textures/block/planks.png": _solid_png(
            (1, 2, 3, 255)
        ),
    }
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    data = _load_npz(dest)
    assert summary.skipped == 1
    assert not [k for k in data.files if k.startswith("quads/minecraft:chest[")]


# Elements-to-quads compiler (plan Task 2).

_FACES = ("down", "up", "north", "south", "west", "east")


def _texkey_stub(files):
    """Build a texkey_for stub resolving face texture refs via a mapping."""

    def resolve(face, textures, ns):
        ref = _resolve_ref(face.get("texture"), textures)
        if ref is None:
            return None
        return files.get(ref)

    return resolve


def _box_element(faces, frm=(0, 0, 0), to=(16, 16, 16)):
    """Build one box element dict with the given per-face dicts."""
    return {"from": list(frm), "to": list(to), "faces": faces}


def test_unit_box_compiles_six_quads():
    faces = {name: {"texture": "#all"} for name in _FACES}
    quads = _compile_elements(
        [_box_element(faces)],
        {"all": "block/stone"},
        "minecraft",
        0,
        0,
        _texkey_stub({"block/stone": "block/stone"}),
    )
    assert quads is not None
    assert quads.dtype == QUAD_DTYPE
    assert len(quads) == 6
    assert sorted(int(t) for t in quads["shade"]) == [0, 1, 2, 3, 4, 5]
    assert not quads["dbl"].any()
    full_uv = [(0, 0), (16, 0), (16, 16), (0, 16)]
    for quad in quads:
        assert quad["texkey"] == "block/stone"
        assert np.allclose(quad["uv"], full_uv)
        assert np.allclose(quad["vtx"], _FACE_GEOM[int(quad["shade"])])


def test_half_box_auto_uv_crops():
    faces = {name: {"texture": "#all"} for name in _FACES}
    quads = _compile_elements(
        [_box_element(faces, to=(16, 8, 16))],
        {"all": "block/stone"},
        "minecraft",
        0,
        0,
        _texkey_stub({"block/stone": "block/stone"}),
    )
    assert quads is not None
    by_tier = {int(q["shade"]): q for q in quads}
    side_uv = [(0, 8), (16, 8), (16, 16), (0, 16)]
    for tier in (TIER_NORTH, TIER_SOUTH, TIER_WEST, TIER_EAST):
        assert np.allclose(by_tier[tier]["uv"], side_uv)
    assert np.allclose(
        by_tier[TIER_SOUTH]["vtx"],
        [(0, 8, 16), (16, 8, 16), (16, 0, 16), (0, 0, 16)],
    )


def test_y_rot_remaps_east_to_south():
    faces = {name: {"texture": f"block/{name}"} for name in _FACES}
    files = {f"block/{name}": f"block/{name}" for name in _FACES}
    quads = _compile_elements(
        [_box_element(faces)], {}, "minecraft", 0, 90, _texkey_stub(files)
    )
    assert quads is not None
    east = next(q for q in quads if q["texkey"] == "block/east")
    assert int(east["shade"]) == TIER_SOUTH
    assert np.allclose(
        east["vtx"], [(16, 16, 16), (0, 16, 16), (0, 0, 16), (16, 0, 16)]
    )
    assert np.allclose(east["uv"], [(0, 0), (16, 0), (16, 16), (0, 16)])


def test_furnace_front_upright_on_south():
    faces = {name: {"texture": "#side"} for name in _FACES}
    faces["north"] = {"texture": "#front"}
    textures = {"front": "block/furnace_front", "side": "block/furnace_side"}
    files = {
        "block/furnace_front": "front_key",
        "block/furnace_side": "side_key",
    }
    quads = _compile_elements(
        [_box_element(faces)], textures, "minecraft", 0, 180, _texkey_stub(files)
    )
    assert quads is not None
    front = [q for q in quads if q["texkey"] == "front_key"]
    assert len(front) == 1
    assert int(front[0]["shade"]) == TIER_SOUTH
    assert np.allclose(
        front[0]["vtx"], [(0, 16, 16), (16, 16, 16), (16, 0, 16), (0, 0, 16)]
    )
    assert np.allclose(front[0]["uv"], [(0, 0), (16, 0), (16, 16), (0, 16)])
    # Unrotated layout with the front directly on the model south face
    # exercises the pinned south auto-UV axis table (u=x, v=16-y).
    faces["north"] = {"texture": "#side"}
    faces["south"] = {"texture": "#front"}
    static = _compile_elements(
        [_box_element(faces)], textures, "minecraft", 0, 0, _texkey_stub(files)
    )
    assert static is not None
    front = [q for q in static if q["texkey"] == "front_key"]
    assert len(front) == 1
    assert int(front[0]["shade"]) == TIER_SOUTH
    assert np.allclose(
        front[0]["vtx"], [(0, 16, 16), (16, 16, 16), (16, 0, 16), (0, 0, 16)]
    )
    assert np.allclose(front[0]["uv"], [(0, 0), (16, 0), (16, 16), (0, 16)])


def test_face_uv_rotation_permutes_rect_corners():
    base = [(4, 8), (12, 8), (12, 16), (4, 16)]
    expected = {
        0: base,
        90: [(4, 16), (4, 8), (12, 8), (12, 16)],
        180: [(12, 16), (4, 16), (4, 8), (12, 8)],
        270: [(12, 8), (12, 16), (4, 16), (4, 8)],
    }
    for rotation, uvs in expected.items():
        quads = _compile_elements(
            [
                _box_element(
                    {
                        "south": {
                            "texture": "#all",
                            "uv": [4, 8, 12, 16],
                            "rotation": rotation,
                        }
                    }
                )
            ],
            {"all": "block/stone"},
            "minecraft",
            0,
            0,
            _texkey_stub({"block/stone": "block/stone"}),
        )
        assert quads is not None
        assert np.allclose(quads[0]["uv"], uvs)


def test_face_uv_rotation_applies_to_auto_uv():
    quads = _compile_elements(
        [_box_element({"south": {"texture": "#all", "rotation": 180}})],
        {"all": "block/stone"},
        "minecraft",
        0,
        0,
        _texkey_stub({"block/stone": "block/stone"}),
    )
    assert quads is not None
    assert np.allclose(quads[0]["uv"], [(16, 16), (0, 16), (0, 0), (16, 0)])


def test_face_uv_rotation_out_of_vanilla_set_skips_face():
    quads = _compile_elements(
        [_box_element({"south": {"texture": "#all", "rotation": 45}})],
        {"all": "block/stone"},
        "minecraft",
        0,
        0,
        _texkey_stub({"block/stone": "block/stone"}),
    )
    assert quads is None


def test_non_parallelogram_face_skipped():
    faces = {name: {"texture": "#all"} for name in _FACES}
    bad = _box_element(faces, frm=(float("nan"), 0, 0))
    quads = _compile_elements(
        [bad, _box_element(faces)],
        {"all": "block/stone"},
        "minecraft",
        0,
        0,
        _texkey_stub({"block/stone": "block/stone"}),
    )
    assert quads is not None
    assert len(quads) == 6
    assert np.isfinite(quads["vtx"]).all()


def test_explicit_uv_used_and_invalid_faces_skipped():
    faces = {
        "south": {"texture": "block/a", "uv": [4, 8, 12, 16]},
        "north": {"texture": "block/a", "uv": [0, 0, 16, 20]},
        "up": {"texture": "#missing"},
    }
    quads = _compile_elements(
        [_box_element(faces)],
        {},
        "minecraft",
        0,
        0,
        _texkey_stub({"block/a": "key_a"}),
    )
    assert quads is not None
    assert len(quads) == 1
    assert int(quads[0]["shade"]) == TIER_SOUTH
    assert np.allclose(quads[0]["uv"], [(4, 8), (12, 8), (12, 16), (4, 16)])


def test_nan_uv_face_skipped():
    faces = {
        "south": {"texture": "block/a", "uv": [float("nan"), 0, 16, 16]},
        "north": {"texture": "block/a"},
    }
    quads = _compile_elements(
        [_box_element(faces)],
        {},
        "minecraft",
        0,
        0,
        _texkey_stub({"block/a": "key_a"}),
    )
    assert quads is not None
    assert len(quads) == 1
    assert int(quads[0]["shade"]) == TIER_NORTH
    assert np.isfinite(quads["uv"]).all()


def test_huge_int_element_skipped():
    faces = {name: {"texture": "#all"} for name in _FACES}
    huge = 10**400
    quads = _compile_elements(
        [
            _box_element(faces, frm=(huge, 0, 0)),
            _box_element({"south": {"texture": "#all", "uv": [0, 0, 16, huge]}}),
            _box_element(faces),
        ],
        {"all": "block/stone"},
        "minecraft",
        0,
        0,
        _texkey_stub({"block/stone": "block/stone"}),
    )
    assert quads is not None
    assert len(quads) == 6
    assert np.isfinite(quads["vtx"]).all()


def test_x_rot_remaps_north_to_down():
    faces = {name: {"texture": f"block/{name}"} for name in _FACES}
    files = {f"block/{name}": f"block/{name}" for name in _FACES}
    quads = _compile_elements(
        [_box_element(faces)], {}, "minecraft", 90, 0, _texkey_stub(files)
    )
    assert quads is not None
    north = next(q for q in quads if q["texkey"] == "block/north")
    assert int(north["shade"]) == TIER_DOWN
    assert np.allclose(north["vtx"], [(16, 0, 0), (0, 0, 0), (0, 0, 16), (16, 0, 16)])
    assert np.allclose(north["uv"], [(0, 0), (16, 0), (16, 16), (0, 16)])


def test_face_geom_matches_unit_cube_quads():
    table = np.stack([_FACE_GEOM[tier] for tier in range(6)])
    assert np.array_equal(table, unit_cube_quads("k")["vtx"])


# npz v3 quads contract (plan Task 3).


def test_npz_v3_roundtrip(tmp_path):
    data = _load_npz(_extract(tmp_path, _stone_entries()))
    assert data["meta/version"].item() == "pack-quads.5"
    assert "quads/minecraft:stone[]" in data.files
    assert "occl/minecraft:stone[]" in data.files
    quads = data["quads/minecraft:stone[]"]
    assert quads.dtype == QUAD_DTYPE
    assert len(quads) == 6
    assert sorted(int(t) for t in quads["shade"]) == [0, 1, 2, 3, 4, 5]
    assert quads["texkey"].tolist() == ["block/stone"] * 6
    assert np.array_equal(quads, sort_quads(quads))
    occl = data["occl/minecraft:stone[]"]
    assert occl.dtype == np.uint8
    assert int(occl.item()) == 1


def test_npz_midwrite_failure_maps_to_cache_write_error(tmp_path, monkeypatch):
    pack_file = _make_pack(tmp_path / "pack.zip", _stone_entries())
    real_zip = zipfile.ZipFile

    class FullDiskZip(real_zip):
        def writestr(self, *args, **kwargs):
            raise OSError("disk full")

    monkeypatch.setattr(zipfile, "ZipFile", FullDiskZip)
    with pytest.raises(_CacheWriteError):
        _extract_pack(pack_file, tmp_path / "out.npz")


def test_occl_requires_unit_cube_and_opaque(tmp_path):
    entries = _stone_entries(
        **{
            "assets/minecraft/blockstates/glass.json": _blockstates("block/glass"),
            "assets/minecraft/models/block/glass.json": _model(
                _all_faces("block/glass")
            ),
            "assets/minecraft/textures/block/glass.png": _solid_png(
                (160, 208, 224, 128)
            ),
            "assets/minecraft/blockstates/slab.json": _blockstates("block/slab"),
            "assets/minecraft/models/block/slab.json": _model(
                _all_faces("block/slab"), to=(16, 8, 16)
            ),
            "assets/minecraft/textures/block/slab.png": _solid_png(
                (110, 84, 52, 255)
            ),
            "assets/minecraft/blockstates/double.json": _blockstates("block/double"),
            "assets/minecraft/models/block/double.json": _two_box_model(
                _all_faces("block/stone")
            ),
        }
    )
    data = _load_npz(_extract(tmp_path, entries))
    assert int(data["occl/minecraft:stone[]"].item()) == 1
    assert int(data["occl/minecraft:glass[]"].item()) == 0
    assert int(data["occl/minecraft:slab[]"].item()) == 0
    assert int(data["occl/minecraft:double[]"].item()) == 0


@pytest.mark.parametrize(
    "rot",
    [{"y": 90}, {"x": 90}, {"x": 180}, {"x": -90}, {"x": 90, "y": 90}],
)
def test_rotated_unit_cube_still_occludes(tmp_path, rot):
    entries = _stone_entries(
        **{
            "assets/minecraft/blockstates/rot.json": _blockstates(
                "block/rot", **rot
            ),
            "assets/minecraft/models/block/rot.json": _model(
                _all_faces("block/stone")
            ),
        }
    )
    data = _load_npz(_extract(tmp_path, entries))
    quads = data["quads/minecraft:rot[]"]
    assert len(quads) == 6
    assert int(data["occl/minecraft:rot[]"].item()) == 1


def test_stub_model_emits_cube_quads(tmp_path):
    data = _load_npz(_extract(tmp_path, _stub_water_entries()))
    quads = data["quads/minecraft:water[]"]
    assert quads.dtype == QUAD_DTYPE
    assert len(quads) == 6
    assert sorted(int(t) for t in quads["shade"]) == [0, 1, 2, 3, 4, 5]
    assert set(quads["texkey"].tolist()) == {"block/water_still@63,118,228"}
    assert (_tex(data, "block/water_still@63,118,228") == (63, 118, 228, 255)).all()


def test_texture_fallback_emits_cube_quads(tmp_path):
    data = _load_npz(_extract(tmp_path, _texture_only_entries()))
    quads = data["quads/minecraft:stone[]"]
    assert quads.dtype == QUAD_DTYPE
    assert len(quads) == 6
    assert set(quads["texkey"].tolist()) == {"block/stone"}
    assert np.array_equal(quads["vtx"], sort_quads(unit_cube_quads("x"))["vtx"])
    assert int(data["occl/minecraft:stone[]"].item()) == 1


def test_quad_fuse_skips_state(tmp_path):
    model = json.dumps(
        {
            "elements": [
                {
                    "from": [0, 0, 0],
                    "to": [16, 16, 16],
                    "faces": _all_faces("block/stone"),
                }
                for _ in range(44)
            ]
        }
    )
    entries = _stone_entries(
        **{
            "assets/minecraft/blockstates/big.json": _blockstates("block/big"),
            "assets/minecraft/models/block/big.json": model,
        }
    )
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    assert summary.blocks == 1
    assert summary.skipped == 1
    data = _load_npz(dest)
    assert "quads/minecraft:stone[]" in data.files
    assert not [k for k in data.files if k.startswith("quads/minecraft:big[")]
    assert not [k for k in data.files if k.startswith("occl/minecraft:big[")]


def test_quad_fuse_boundary_256_kept(tmp_path):
    elements = [
        {"from": [0, 0, 0], "to": [16, 16, 16], "faces": _all_faces("block/stone")}
        for _ in range(42)
    ]
    elements.append(
        {
            "from": [0, 0, 0],
            "to": [16, 16, 16],
            "faces": {
                name: {"texture": "block/stone"}
                for name in ("down", "up", "north", "south")
            },
        }
    )
    entries = _stone_entries(
        **{
            "assets/minecraft/blockstates/edge.json": _blockstates("block/edge"),
            "assets/minecraft/models/block/edge.json": json.dumps(
                {"elements": elements}
            ),
        }
    )
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    assert summary.blocks == 2
    assert summary.skipped == 0
    data = _load_npz(dest)
    assert len(data["quads/minecraft:edge[]"]) == 256


# Non-box elements: rotation, planes, dbl, shade (spec 5.1 / 3 / 11).


def _element(frm, to, faces, rotation=None, shade=None):
    """Build one element dict with optional rotation and shade overrides."""
    element = {"from": list(frm), "to": list(to), "faces": faces}
    if rotation is not None:
        element["rotation"] = rotation
    if shade is not None:
        element["shade"] = shade
    return element


def _cross_rotation(angle, rescale=True, origin=(8, 8, 8), axis="y"):
    """Vanilla-style cross element rotation: 45 degrees, optional rescale."""
    return {
        "origin": list(origin),
        "axis": axis,
        "angle": angle,
        "rescale": rescale,
    }


def _cross_faces():
    return {"north": {"texture": "#cross"}, "south": {"texture": "#cross"}}


def _compile_one(element, texkey="block/grass"):
    return _compile_elements(
        [element],
        {"cross": texkey},
        "minecraft",
        0,
        0,
        _texkey_stub({texkey: texkey}),
    )


def _corner_set(quads):
    """Sorted rounded unique vertex tuples across all quads."""
    return sorted(
        {
            tuple(round(float(v), 4) for v in point)
            for quad in quads
            for point in quad["vtx"]
        }
    )


def test_cross_element_compiles_rotated_double_sided():
    quads = _compile_one(
        _element(
            (0.8, 0, 8), (15.2, 16, 8), _cross_faces(),
            rotation=_cross_rotation(45),
        )
    )
    assert quads is not None
    assert len(quads) == 2
    assert quads["dbl"].all()
    assert (quads["shade"] == TIER_UP).all()
    assert _corner_set(quads) == sorted(
        [
            (0.8, 0.0, 15.2),
            (15.2, 0.0, 0.8),
            (15.2, 16.0, 0.8),
            (0.8, 16.0, 15.2),
        ]
    )


def test_cross_element_negative_angle_other_diagonal():
    quads = _compile_one(
        _element(
            (0.8, 0, 8), (15.2, 16, 8), _cross_faces(),
            rotation=_cross_rotation(-45),
        )
    )
    assert quads is not None
    assert _corner_set(quads) == sorted(
        [
            (0.8, 0.0, 0.8),
            (15.2, 0.0, 15.2),
            (15.2, 16.0, 15.2),
            (0.8, 16.0, 0.8),
        ]
    )


def test_flat_element_marks_dbl_up_tier():
    quads = _compile_one(
        _element((0, 0, 8), (16, 16, 8), _cross_faces())
    )
    assert quads is not None
    assert len(quads) == 2
    assert quads["dbl"].all()
    assert (quads["shade"] == TIER_UP).all()
    assert (quads["vtx"][..., 2] == 8.0).all()
    assert np.allclose(quads["uv"], [[(0, 0), (16, 0), (16, 16), (0, 16)]] * 2)


def test_flat_element_skips_degenerate_faces():
    faces = {**_cross_faces(), "up": {"texture": "#cross"}}
    quads = _compile_one(_element((0, 0, 8), (16, 16, 8), faces))
    assert quads is not None
    assert len(quads) == 2


def test_box_shade_false_renders_constant():
    faces = {name: {"texture": "#cross"} for name in _FACES}
    quads = _compile_one(_element((0, 0, 0), (16, 16, 16), faces, shade=False))
    assert quads is not None
    assert len(quads) == 6
    assert not quads["dbl"].any()
    assert (quads["shade"] == TIER_UP).all()


def test_rescale_false_shrinks_rotated_plane():
    quads = _compile_one(
        _element(
            (0.8, 0, 8), (15.2, 16, 8), _cross_faces(),
            rotation=_cross_rotation(45, rescale=False),
        )
    )
    assert quads is not None
    half = 7.2 * 2 ** -0.5
    assert _corner_set(quads) == sorted(
        [
            (round(8 - half, 4), 0.0, round(8 + half, 4)),
            (round(8 + half, 4), 0.0, round(8 - half, 4)),
            (round(8 + half, 4), 16.0, round(8 - half, 4)),
            (round(8 - half, 4), 16.0, round(8 + half, 4)),
        ]
    )


def test_rotated_box_element_is_double_sided():
    faces = {name: {"texture": "#cross"} for name in _FACES}
    quads = _compile_one(
        _element(
            (0, 0, 0), (16, 16, 16), faces,
            rotation=_cross_rotation(45, rescale=False),
        )
    )
    assert quads is not None
    assert len(quads) == 6
    assert quads["dbl"].all()
    assert (quads["shade"] == TIER_UP).all()
    diag = 8.0 * 2 ** 0.5
    hull = sorted(
        {
            (round(8 + x, 4), y, round(8 + z, 4))
            for x, z in ((diag, 0.0), (0.0, -diag), (0.0, diag), (-diag, 0.0))
            for y in (0.0, 16.0)
        }
    )
    assert _corner_set(quads) == hull


def test_malformed_element_rotation_skips_element():
    for rotation in ({"angle": 30}, {"axis": "q", "angle": 45}, "nope", 5):
        element = _element(
            (0.8, 0, 8), (15.2, 16, 8), _cross_faces(), rotation=rotation
        )
        assert _compile_one(element) is None, rotation


def test_line_element_skipped():
    quads = _compile_one(
        _element((8, 0, 8), (8, 16, 8), _cross_faces())
    )
    assert quads is None


def test_element_rotation_origin_matters():
    quads = _compile_one(
        _element(
            (0.8, 0, 8), (15.2, 16, 8), _cross_faces(),
            rotation=_cross_rotation(45, origin=(0, 0, 0)),
        )
    )
    assert quads is not None
    assert _corner_set(quads) == sorted(
        [
            (8.8, 0.0, 7.2),
            (23.2, 0.0, -7.2),
            (23.2, 16.0, -7.2),
            (8.8, 16.0, 7.2),
        ]
    )


# Texture-direct fallback absorbed by the builtin shape table (plan Task 6
# adjudication: builtin geometry wins, the bare cube only covers unknown
# families).


def test_texture_fallback_uses_builtin_cross_geometry(tmp_path):
    # Cross plants are a builtin family: no extraction quads (compose's
    # state-aware builtin renders them), texture registered for backfill.
    entries = {
        "assets/minecraft/textures/block/short_grass.png": _solid_png(
            (90, 160, 70, 255)
        )
    }
    data = _load_npz(_extract(tmp_path, entries))
    assert not [k for k in data.files if k.startswith("quads/minecraft:short_grass[")]
    tinted = _tex(data, "block/short_grass@145,189,89")
    factor = np.array([145, 189, 89, 255], np.uint16)
    want = (np.array([(90, 160, 70, 255)], np.uint16) * factor // 255).astype(
        np.uint8
    )
    assert (tinted == want[0]).all()


def test_texture_fallback_builtin_slab_keeps_shape_not_occl(tmp_path):
    entries = {
        "assets/minecraft/textures/block/oak_slab.png": _solid_png(
            (110, 84, 52, 255)
        )
    }
    data = _load_npz(_extract(tmp_path, entries))
    assert not [k for k in data.files if k.startswith("quads/minecraft:oak_slab[")]
    assert (_tex(data, "block/oak_slab") == (110, 84, 52, 255)).all()


def test_texture_fallback_unknown_family_stays_cube(tmp_path):
    entries = {
        "assets/minecraft/textures/block/mud_like.png": _solid_png(
            (120, 85, 60, 255)
        )
    }
    data = _load_npz(_extract(tmp_path, entries))
    quads = data["quads/minecraft:mud_like[]"]
    assert len(quads) == 6
    assert not quads["dbl"].any()
    assert int(data["occl/minecraft:mud_like[]"].item()) == 1


# Multipart blockstates (spec gap: only all-unconditional multiparts carry
# pack geometry; conditional ones defer to builtin shapes / magenta).


def _face_model(stem, texture):
    return json.dumps(
        {
            "elements": [
                {
                    "from": [0, 0, 0],
                    "to": [16, 16, 16],
                    "faces": {"south": {"texture": texture}},
                }
            ]
        }
    )


def _multipart_entries(multipart):
    return {
        "assets/minecraft/blockstates/gate.json": json.dumps(
            {"multipart": multipart}
        ),
        "assets/minecraft/models/block/post.json": _face_model(
            "post", "block/post"
        ),
        "assets/minecraft/models/block/arm.json": _face_model(
            "arm", "block/arm"
        ),
        "assets/minecraft/textures/block/post.png": _solid_png(
            (1, 2, 3, 255)
        ),
        "assets/minecraft/textures/block/arm.png": _solid_png((4, 5, 6, 255)),
    }


def test_multipart_conditional_parts_merge_per_state(tmp_path):
    # Lazy when-merge contract (Blockscope select_variants shape): the
    # npz stores the raw condition tree plus per-part geometry; pack.py
    # evaluates the tree against the live properties at lookup.
    entries = _multipart_entries(
        [
            {"apply": {"model": "block/post"}},
            {"when": {"north": "true"}, "apply": {"model": "block/arm"}},
        ]
    )
    pack_file = _make_pack(tmp_path / "pack.zip", entries)
    _extract(pack_file, tmp_path) if False else _extract(tmp_path, entries)
    data = _load_npz(tmp_path / "out.npz")
    record = json.loads(str(data["parts/minecraft:gate"].item()))
    assert [p["when"] for p in record["parts"]] == [None, {"north": "true"}]
    textures = load_pack(pack_file)
    off = textures.quads("minecraft:gate", {"north": "false"})
    assert sorted(str(k) for k in off["texkey"]) == ["block/post"]
    on = textures.quads("minecraft:gate", {"north": "true"})
    assert sorted(str(k) for k in on["texkey"]) == ["block/arm", "block/post"]


def test_multipart_all_conditional_evaluates_per_state(tmp_path):
    entries = _multipart_entries(
        [{"when": {"north": "true"}, "apply": {"model": "block/arm"}}]
    )
    pack_file = _make_pack(tmp_path / "pack.zip", entries)
    _extract(tmp_path, entries)
    textures = load_pack(pack_file)
    on = textures.quads("minecraft:gate", {"north": "true"})
    assert sorted(str(k) for k in on["texkey"]) == ["block/arm"]
    # A state matching no condition has no part geometry and no [] key.
    assert textures.quads("minecraft:gate", {"north": "false"}) is None


def test_multipart_unconditional_parts_union(tmp_path):
    entries = _multipart_entries(
        [{"apply": {"model": "block/post"}}, {"apply": {"model": "block/arm"}}]
    )
    pack_file = _make_pack(tmp_path / "pack.zip", entries)
    _extract(tmp_path, entries)
    textures = load_pack(pack_file)
    quads = textures.quads("minecraft:gate", {})
    assert sorted({str(k) for k in quads["texkey"]}) == [
        "block/arm",
        "block/post",
    ]


def test_multipart_unconditional_apply_list_union(tmp_path):
    entries = _multipart_entries(
        [
            {
                "apply": [
                    {"model": "block/post"},
                    {"model": "block/arm", "x": 90},
                ]
            }
        ]
    )
    pack_file = _make_pack(tmp_path / "pack.zip", entries)
    _extract(tmp_path, entries)
    textures = load_pack(pack_file)
    quads = textures.quads("minecraft:gate", {})
    assert sorted({str(k) for k in quads["texkey"]}) == [
        "block/arm",
        "block/post",
    ]


def test_multipart_single_unconditional_part(tmp_path):
    entries = _multipart_entries([{"apply": {"model": "block/post"}}])
    pack_file = _make_pack(tmp_path / "pack.zip", entries)
    _extract(tmp_path, entries)
    textures = load_pack(pack_file)
    quads = textures.quads("minecraft:gate", {})
    assert set(str(k) for k in quads["texkey"]) == {"block/post"}


# Round-2 layering: blocks the builtin table knows must never get
# extraction-side geometry (it would shadow compose's state-aware builtin),
# and their texture resolves through the spec 5.3 chain: the block's own
# model textures first, then textures/block/<stem>.png.


def _fence_pack_entries(**extra):
    model = json.dumps(
        {
            "textures": {"planks": "block/oak_planks"},
            "elements": [
                {
                    "from": [0, 0, 0],
                    "to": [16, 16, 16],
                    "faces": {"south": {"texture": "#planks"}},
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
                ]
            }
        ),
        "assets/minecraft/models/block/fence_post.json": model,
        "assets/minecraft/models/block/fence_side.json": model,
        "assets/minecraft/textures/block/oak_planks.png": _solid_png(
            (168, 90, 50, 255)
        ),
    }
    entries.update(extra)
    return entries


def test_vanilla_fence_geometry_comes_from_multipart_merge(tmp_path):
    entries = _fence_pack_entries(
        **{
            "assets/minecraft/textures/block/oak_fence.png": _solid_png(
                (9, 9, 9, 255)
            )
        }
    )
    pack_file = _make_pack(tmp_path / "pack.zip", entries)
    summary = _extract_pack(pack_file, tmp_path / "out.npz")
    assert summary.skipped == 0
    textures = load_pack(pack_file)
    # The model's own planks texture resolves inline; the same-name file
    # never participates. north=false renders the post model alone.
    alone = textures.quads("minecraft:oak_fence", {"north": "false"})
    assert [str(k) for k in alone["texkey"]] == ["block/oak_planks"]
    linked = textures.quads("minecraft:oak_fence", {"north": "true"})
    assert sorted(str(k) for k in linked["texkey"]) == [
        "block/oak_planks"
    ] * 2


def test_vanilla_fence_without_same_name_texture_still_resolves(tmp_path):
    entries = _fence_pack_entries()
    pack_file = _make_pack(tmp_path / "pack.zip", entries)
    _extract(tmp_path, entries)
    textures = load_pack(pack_file)
    quads = textures.quads("minecraft:oak_fence", {"north": "false"})
    assert [str(k) for k in quads["texkey"]] == ["block/oak_planks"]


def test_broken_model_family_keeps_same_name_texture(tmp_path):
    entries = {
        "assets/minecraft/blockstates/oak_stairs.json": _blockstates(
            "block/missing_model"
        ),
        "assets/minecraft/textures/block/oak_stairs.png": _solid_png(
            (110, 84, 52, 255)
        ),
    }
    data = _load_npz(_extract(tmp_path, entries))
    assert not [k for k in data.files if k.startswith("quads/minecraft:oak_stairs[")]
    assert (_tex(data, "block/oak_stairs") == (110, 84, 52, 255)).all()


def test_builtin_entity_parent_registers_model_texture(tmp_path):
    entries = {
        "assets/minecraft/blockstates/chest.json": _blockstates("block/chest"),
        "assets/minecraft/models/block/chest.json": json.dumps(
            {"parent": "builtin/entity", "textures": {"particle": "block/planks"}}
        ),
        "assets/minecraft/textures/block/planks.png": _solid_png((1, 2, 3, 255)),
    }
    data = _load_npz(_extract(tmp_path, entries))
    assert not [k for k in data.files if k.startswith("quads/minecraft:chest[")]
    assert (_tex(data, "block/chest") == (1, 2, 3, 255)).all()


def test_texture_fallback_family_registers_texture_without_quads(tmp_path):
    entries = {
        "assets/minecraft/textures/block/short_grass.png": _solid_png(
            (98, 158, 68, 255)
        )
    }
    data = _load_npz(_extract(tmp_path, entries))
    assert not [k for k in data.files if k.startswith("quads/minecraft:short_grass[")]
    # v1 dedup: tinted blocks register only the @tint array key.
    tinted = _tex(data, "block/short_grass@145,189,89")
    assert (tinted == (55, 117, 23, 255)).all()


def test_wire_dust_textures_registered_with_power_tints(tmp_path):
    entries = {
        "assets/minecraft/textures/block/redstone_dust_dot.png": _solid_png(
            (220, 220, 220, 255)
        ),
        "assets/minecraft/textures/block/redstone_dust_line0.png": _solid_png(
            (220, 220, 220, 255)
        ),
    }
    data = _load_npz(_extract(tmp_path, entries))
    for key in (
        "block/redstone_dust_dot@76,0,0",
        "block/redstone_dust_dot@255,26,26",
        "block/redstone_dust_line0@171,14,14",
    ):
        assert f"tex/{key}" in data.files
    assert tuple(data["tex/block/redstone_dust_dot@255,26,26"][0, 0]) == (
        220,
        22,
        22,
        255,
    )


def test_multipart_when_conditions_merge_per_state(tmp_path):
    entries = {
        "assets/minecraft/blockstates/post.json": json.dumps(
            {
                "multipart": [
                    {"apply": {"model": "block/post"}},
                    {"apply": {"model": "block/arm"}, "when": {"north": "true"}},
                    {
                        "apply": {"model": "block/up"},
                        "when": {"OR": [{"east": "side"}, {"east": "up"}]},
                    },
                ]
            }
        ),
        "assets/minecraft/models/block/post.json": _model(_all_faces("block/post")),
        "assets/minecraft/models/block/arm.json": _model(_all_faces("block/arm")),
        "assets/minecraft/models/block/up.json": _model(_all_faces("block/up")),
        "assets/minecraft/textures/block/post.png": _solid_png((9, 9, 9, 255)),
        "assets/minecraft/textures/block/arm.png": _solid_png((8, 8, 8, 255)),
        "assets/minecraft/textures/block/up.png": _solid_png((7, 7, 7, 255)),
    }
    pack_file = _make_pack(tmp_path / "pack.zip", entries)
    _extract(tmp_path, entries)
    textures = load_pack(pack_file)
    both = textures.quads("minecraft:post", {"north": "true", "east": "side"})
    assert sorted({str(k) for k in both["texkey"]}) == [
        "block/arm",
        "block/post",
        "block/up",
    ]
    none_east = textures.quads(
        "minecraft:post", {"north": "true", "east": "none"}
    )
    assert sorted({str(k) for k in none_east["texkey"]}) == [
        "block/arm",
        "block/post",
    ]


def test_wire_multipart_enumerates_power_for_tint(tmp_path):
    wire_state = json.dumps(
        {
            "multipart": [
                {"apply": {"model": "block/wire"}},
                {
                    "apply": {"model": "block/wire_up"},
                    "when": {"north": "side|up"},
                },
            ]
        }
    )
    wire_model = json.dumps(
        {
            "elements": [
                {
                    "from": [0, 0, 0],
                    "to": [16, 0, 16],
                    "faces": {"up": {"texture": "block/wire", "tintindex": 0}},
                }
            ]
        }
    )
    entries = {
        "assets/minecraft/blockstates/redstone_wire.json": wire_state,
        "assets/minecraft/models/block/wire.json": wire_model,
        "assets/minecraft/models/block/wire_up.json": wire_model,
        "assets/minecraft/textures/block/wire.png": _solid_png(
            (220, 220, 220, 255)
        ),
    }
    pack_file = _make_pack(tmp_path / "pack.zip", entries)
    _extract(tmp_path, entries)
    textures = load_pack(pack_file)
    mid = textures.quads("minecraft:redstone_wire", {"power": "8", "north": "none"})
    assert [str(k) for k in mid["texkey"]] == ["block/wire@171,14,14"]
    zero = textures.quads("minecraft:redstone_wire", {"power": "0", "north": "none"})
    assert [str(k) for k in zero["texkey"]] == ["block/wire@76,0,0"]


# Native-resolution textures: pack pixels reach the npz unsampled.


def test_native_resolution_texture_preserved(tmp_path):
    dest = _extract(tmp_path, _stone_entries(
        **{"assets/minecraft/textures/block/stone.png":
           _solid_png((200, 30, 30, 255), size=(32, 32))}
    ))
    data = _load_npz(dest)
    tex = _tex(data, "block/stone")
    assert tex.shape == (32, 32, 4)
    assert tex.dtype == np.uint8


def test_mcmeta_square_animation_first_frame(tmp_path):
    entries = _stone_entries(
        **{
            "assets/minecraft/textures/block/stone.png": _strip_png(
                (200, 30, 30, 255), (10, 200, 10, 255)
            ),
            "assets/minecraft/textures/block/stone.png.mcmeta":
                '{"animation":{"frametime":2}}',
        }
    )
    data = _load_npz(_extract(tmp_path, entries))
    tex = _tex(data, "block/stone")
    assert tex.shape == (16, 16, 4)
    assert (tex == (200, 30, 30, 255)).all()


def test_mcmeta_without_animation_key_kept_whole(tmp_path):
    entries = _stone_entries(
        **{
            "assets/minecraft/textures/block/stone.png":
                _solid_png((200, 30, 30, 255), size=(16, 8)),
            "assets/minecraft/textures/block/stone.png.mcmeta":
                '{"texture":{}}',
        }
    )
    data = _load_npz(_extract(tmp_path, entries))
    assert _tex(data, "block/stone").shape == (8, 16, 4)


def test_non_square_non_animated_kept_whole(tmp_path):
    entries = _stone_entries(
        **{"assets/minecraft/textures/block/stone.png":
           _solid_png((200, 30, 30, 255), size=(16, 8))}
    )
    data = _load_npz(_extract(tmp_path, entries))
    tex = _tex(data, "block/stone")
    assert tex.shape == (8, 16, 4)
    assert (tex == (200, 30, 30, 255)).all()


def test_oversize_side_rejected(tmp_path):
    dest = _extract(tmp_path, _stone_entries(
        **{"assets/minecraft/textures/block/stone.png":
           _solid_png((200, 30, 30, 255), size=(2049, 64))}
    ))
    data = _load_npz(dest)
    assert not any(key.startswith("tex/") for key in data.files)
    assert not any(key.startswith("quads/") for key in data.files)


def test_budget_exhaustion_skips_later_textures(tmp_path, monkeypatch):
    from litematic.render.pack import texture as texture_mod

    monkeypatch.setattr(texture_mod._Budget, "MAX_BYTES", 16 * 16 * 4)
    entries = _stone_entries(
        **{
            "assets/minecraft/blockstates/b.json":
                _blockstates("block/b"),
            "assets/minecraft/models/block/b.json":
                _model(_all_faces("block/b")),
            "assets/minecraft/textures/block/b.png":
                _solid_png((10, 200, 10, 255)),
        }
    )
    dest = _extract(tmp_path, entries)
    data = _load_npz(dest)
    # Sorted state files extract "b" first; it claims the entire budget.
    assert "tex/block/b" in data.files
    assert "tex/block/stone" not in data.files
    assert "quads/minecraft:stone[]" not in data.files


def test_cullface_mapped_through_variant_rotation(tmp_path):
    faces = {
        name: {"texture": "#all"}
        for name in ("down", "up", "south", "west", "east")
    }
    faces["north"] = {"texture": "#all", "cullface": "north"}
    entries = _stone_entries(
        **{
            "assets/minecraft/models/block/stone.json": json.dumps(
                {
                    "textures": {"all": "block/stone"},
                    "elements": [
                        {"from": [0, 0, 0], "to": [16, 16, 16], "faces": faces}
                    ],
                }
            ),
            "assets/minecraft/blockstates/rot.json": _blockstates(
                "block/stone", y=180
            ),
        }
    )
    data = _load_npz(_extract(tmp_path, entries))
    quads = data["quads/minecraft:rot[]"]
    south = next(q for q in quads if int(q["shade"]) == TIER_SOUTH)
    assert int(south["cull"]) == TIER_SOUTH  # y180 rotates the cullface
    for tier, name in (
        (TIER_DOWN, "block/stone"),
        (TIER_UP, "block/stone"),
        (TIER_EAST, "block/stone"),
    ):
        quad = next(q for q in quads if int(q["shade"]) == tier)
        assert int(quad["cull"]) == CULL_NONE


def test_missing_cullface_defaults_to_none(tmp_path):
    data = _load_npz(_extract(tmp_path, _stone_entries()))
    quads = data["quads/minecraft:stone[]"]
    assert (quads["cull"] == CULL_NONE).all()


def test_pack_quads4_cache_rebuilds(tmp_path):
    pack_file = _make_pack(tmp_path / "pack.zip", _stone_entries())
    dest = tmp_path / "out.npz"
    _extract_pack(pack_file, dest)
    data = dict(np.load(dest, allow_pickle=False))
    data["meta/version"] = np.array("pack-quads.4")
    cache = tmp_path / "pack.litematic-tex.npz"
    np.savez(cache, **data)
    textures = load_pack(pack_file)
    assert textures.version == "pack-quads.5"
