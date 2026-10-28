"""Pack extraction contract: blockstates→models→textures→npz (state-keyed)."""

import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from litematic.render.errors import RenderError
from litematic.render.extract import ExtractSummary, _extract_pack


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


def _model(faces, textures=None, parent=None):
    model = {}
    if parent is not None:
        model["parent"] = parent
    if textures is not None:
        model["textures"] = textures
    model["elements"] = [{"from": [0, 0, 0], "to": [16, 16, 16], "faces": faces}]
    return json.dumps(model)


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


def _ref(data, block, state, face):
    return data[f"map/{block}[{state}]/{face}"].item()


def _tex(data, key):
    return data[f"tex/{key}"]


def test_plain_block_stateless_entry(tmp_path):
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", _stone_entries()), dest)
    assert isinstance(summary, ExtractSummary)
    assert summary.blocks == 1
    assert summary.skipped == 0
    data = _load_npz(dest)
    assert _ref(data, "minecraft:stone", "", "up") == "block/stone"
    tex = _tex(data, "block/stone")
    assert tex.shape == (16, 16, 4)
    assert tex.dtype == np.uint8
    assert (tex == (200, 30, 30, 255)).all()
    for face in ("down", "up", "north", "south", "west", "east"):
        assert f"map/minecraft:stone[]/{face}" in data.files
    assert data["meta/version"].item() == "pack"


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
    assert _ref(data, "minecraft:pillar", "", "up") == "block/end_tex"
    assert _ref(data, "minecraft:pillar", "", "east") == "block/side_tex"
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
    assert _ref(data, "minecraft:child", "", "up") == "block/new"
    assert all(
        _ref(data, "minecraft:child", "", face) == "block/new"
        for face in ("up", "down", "north", "south", "west", "east")
    )


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
    assert _ref(data, "minecraft:front", "", "east") == "block/north_tex"
    assert _ref(data, "minecraft:front", "", "south") == "block/east_tex"
    assert _ref(data, "minecraft:front", "", "up") == "block/up_tex"


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
    assert _ref(data, "minecraft:furnace", "facing=south", "south") == "block/front_tex"
    assert _ref(data, "minecraft:furnace", "facing=north", "north") == "block/front_tex"
    assert _ref(data, "minecraft:furnace", "facing=north", "east") == "block/side_tex"


def test_state_key_pairs_sorted_by_property_name(tmp_path):
    entries = _stone_entries()
    entries["assets/minecraft/blockstates/unsorted.json"] = _blockstates_variants(
        {"shape=straight,half=bottom,facing=east": {"model": "block/stone"}}
    )
    data = _load_npz(_extract(tmp_path, entries))
    key = "map/minecraft:unsorted[facing=east,half=bottom,shape=straight]/up"
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
    assert _ref(data, "minecraft:grass_block", "", "up") == "block/grass@145,189,89"
    assert (_tex(data, "block/grass@145,189,89") == (145, 189, 89, 255)).all()
    assert _ref(data, "minecraft:grass_block", "", "south") == "block/grass"
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
    low = _tex(data, _ref(data, "minecraft:redstone_wire", "power=0", "up"))
    high = _tex(data, _ref(data, "minecraft:redstone_wire", "power=15", "up"))
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
    tex = _tex(data, _ref(data, f"minecraft:{block}", "", "up"))
    assert (tex[0, 0] == (*color, 255)).all()


def test_animation_strip_takes_first_frame(tmp_path):
    entries = _stone_entries()
    entries["assets/minecraft/textures/block/stone.png"] = _strip_png(
        (200, 0, 0, 255), (0, 0, 200, 255)
    )
    data = _load_npz(_extract(tmp_path, entries))
    assert (_tex(data, "block/stone") == (200, 0, 0, 255)).all()


def test_oversize_texture_downsampled_to_16(tmp_path):
    entries = _stone_entries()
    entries["assets/minecraft/textures/block/stone.png"] = _quadrant_png()
    data = _load_npz(_extract(tmp_path, entries))
    tex = _tex(data, "block/stone")
    assert tex.shape == (16, 16, 4)
    assert (tex[0, 0][:3] == (255, 0, 0)).all()
    assert (tex[15, 15][:3] == (0, 255, 0)).all()


def test_no_face_model_skipped_and_counted(tmp_path):
    entries = _stone_entries()
    entries["assets/minecraft/blockstates/torch.json"] = _blockstates("block/torch")
    entries["assets/minecraft/models/block/torch.json"] = _model(
        {"up": {"texture": "block/stone"}}
    )
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    assert summary.blocks == 1
    assert summary.skipped == 1
    assert not any(f.startswith("map/minecraft:torch") for f in _load_npz(dest).files)


def test_malformed_model_skips_block(tmp_path):
    entries = _stone_entries()
    entries["assets/minecraft/blockstates/bad.json"] = _blockstates("block/bad")
    entries["assets/minecraft/models/block/bad.json"] = "{not json"
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    assert summary.blocks == 1
    assert summary.skipped == 1
    data = _load_npz(dest)
    assert "map/minecraft:stone[]/up" in data.files
    assert not any(f.startswith("map/minecraft:bad") for f in data.files)


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
    assert _ref(data, "minecraft:chained", "", "up") == "block/deep"
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
    assert _ref(data, "minecraft:namespaced", "", "up") == "block/prefixed"
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


def test_deterministic_output_bytes(tmp_path):
    pack = _make_pack(tmp_path / "pack.zip", _stone_entries())
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    _extract_pack(pack, first)
    _extract_pack(pack, second)
    assert first.read_bytes() == second.read_bytes()


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
        _tex(data, _ref(data, "alpha:shared", "", "up")) == (10, 10, 10, 255)
    ).all()
    assert (
        _tex(data, _ref(data, "beta:shared", "", "up")) == (20, 20, 20, 255)
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
    for face in ("up", "down", "north", "south", "west", "east"):
        assert _ref(data, "minecraft:stone", "", face) == "block/stone"
    assert (
        _tex(data, "block/stone") == (200, 30, 30, 255)
    ).all()
    assert "map/modmenu:mods_button[]/up" not in data.files
    assert "map/minecraft:deep[]/up" not in data.files
    assert "map/minecraft:stick[]/up" not in data.files
    assert "map/minecraft:UPPER[]/up" not in data.files


def test_texture_fallback_tints_by_block_name(tmp_path):
    data = _load_npz(_extract(tmp_path, _texture_only_entries()))
    key = _ref(data, "minecraft:grass_block", "", "up")
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
    assert "map/minecraft:dirt[]/up" in data.files
    assert summary.blocks == 2
    assert summary.skipped == 0


def test_foreign_namespace_texture_fallback(tmp_path):
    entries = {
        "assets/create/textures/block/limestone.png": _solid_png(
            (180, 180, 170, 255)
        )
    }
    data = _load_npz(_extract(tmp_path, entries))
    assert _ref(data, "create:limestone", "", "up") == "create:block/limestone"


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


def test_stub_model_falls_back_to_particle_texture(tmp_path):
    data = _load_npz(_extract(tmp_path, _stub_water_entries()))
    for face in ("up", "east", "south"):
        key = _ref(data, "minecraft:water", "", face)
        assert key == "block/water_still@63,118,228"
    assert (
        _tex(data, "block/water_still@63,118,228") == (63, 118, 228, 255)
    ).all()


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
    assert not [k for k in data.files if k.startswith("map/minecraft:portal[")]


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
    assert _ref(data, "minecraft:flow", "", "up") == "block/still"


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
    assert not [k for k in data.files if k.startswith("map/minecraft:chest[")]
