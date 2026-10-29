"""Block-entity-shaped compilation: chest geometry from entity textures."""

import io
import json
import zipfile

import numpy as np
from PIL import Image

from litematic.render.pack.entities import chest_quads
from litematic.render.pack.extract import _extract_pack
from litematic.render.quads import CULL_NONE, QUAD_DTYPE, TIER_SOUTH


def _solid_png(color, size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, "PNG")
    return buf.getvalue()


def _make_pack(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _load_npz(dest):
    return np.load(dest, allow_pickle=False)


def _spy(refs):
    def resolve(ref):
        refs.append(ref)
        return f"texkey:{ref}"

    return resolve


def test_single_chest_compiles_18_quads():
    refs = []
    quads = chest_quads(
        "minecraft:chest", "facing=north,type=single", _spy(refs)
    )
    assert refs == ["entity/chest/normal"]
    assert quads is not None
    assert len(quads) == 18
    assert quads.dtype == QUAD_DTYPE
    assert not quads["dbl"].any()
    assert sorted(set(int(t) for t in quads["shade"])) == [0, 1, 2, 3, 4, 5]
    assert set(str(k) for k in quads["texkey"]) == {"texkey:entity/chest/normal"}


def test_lock_protrudes_on_faced_side():
    cases = {
        "north": (2, 0, "low"),
        "south": (2, 0, "high"),
        "east": (0, 2, "high"),
        "west": (0, 2, "low"),
    }
    for facing, (axis, cross, side) in cases.items():
        quads = chest_quads(
            "minecraft:chest", f"facing={facing},type=single", _spy([])
        )
        assert quads is not None
        lo = quads["vtx"][..., axis].min(axis=1)
        hi = quads["vtx"][..., axis].max(axis=1)
        cross_lo = quads["vtx"][..., cross].min(axis=1)
        cross_hi = quads["vtx"][..., cross].max(axis=1)
        if side == "low":
            lock = quads[
                (hi <= 1.0) & (cross_lo >= 7.0) & (cross_hi <= 9.0)
            ]
        else:
            lock = quads[
                (lo >= 15.0) & (cross_lo >= 7.0) & (cross_hi <= 9.0)
            ]
        assert len(lock) == 6, facing


def test_body_south_face_uv_samples_atlas_front_region():
    quads = chest_quads(
        "minecraft:chest", "facing=south,type=single", _spy([])
    )
    assert quads is not None
    body_south = [
        quad
        for quad in quads
        if int(quad["shade"]) == TIER_SOUTH
        and quad["vtx"][..., 1].min() == 0.0
    ]
    assert len(body_south) == 1
    quad = body_south[0]
    assert np.allclose(
        quad["vtx"], [(15, 0, 15), (15, 10, 15), (1, 10, 15), (1, 0, 15)]
    )
    # The body front maps to the 64-space atlas rect (42,33)-(56,43).
    assert np.allclose(
        quad["uv"], [(10.5, 8.25), (10.5, 10.75), (14, 10.75), (14, 8.25)]
    )


def test_half_chests_split_and_lock_at_outer_edge():
    left = chest_quads("minecraft:chest", "facing=south,type=left", _spy([]))
    right = chest_quads("minecraft:chest", "facing=south,type=right", _spy([]))
    assert left is not None and len(left) == 18
    assert right is not None and len(right) == 18
    assert left["vtx"][..., 0].min() == 0.0
    assert left["vtx"][..., 0].max() == 15.0
    assert right["vtx"][..., 0].min() == 1.0
    assert right["vtx"][..., 0].max() == 16.0
    left_lock = left[
        (left["vtx"][..., 0].max(axis=1) <= 1.0)
        & (left["vtx"][..., 2].min(axis=1) >= 15.0)
    ]
    right_lock = right[
        (right["vtx"][..., 0].min(axis=1) >= 15.0)
        & (right["vtx"][..., 2].min(axis=1) >= 15.0)
    ]
    assert len(left_lock) == 6
    assert len(right_lock) == 6


def test_texture_ref_per_block_and_kind():
    refs = []
    chest_quads("minecraft:trapped_chest", "facing=north,type=single", _spy(refs))
    chest_quads("minecraft:ender_chest", "facing=north,type=left", _spy(refs))
    assert refs == ["entity/chest/trapped", "entity/chest/ender_left"]


def test_unknown_block_or_kind_rejected():
    assert chest_quads("minecraft:barrel", "", _spy([])) is None
    assert chest_quads("minecraft:chest", "type=weird", _spy([])) is None
    assert chest_quads("minecraft:chest", "", _spy([])) is not None


def _chest_entries():
    faces = {
        "facing=north,type=single,waterlogged=false": {"model": "block/chest"},
        "facing=south,type=single,waterlogged=false": {"model": "block/chest"},
    }
    return {
        "assets/minecraft/blockstates/chest.json": json.dumps(
            {"variants": faces}
        ),
        "assets/minecraft/models/block/chest.json": json.dumps(
            {"parent": "builtin/entity", "textures": {"particle": "block/planks"}}
        ),
        "assets/minecraft/textures/block/planks.png": _solid_png(
            (160, 130, 78, 255), size=(16, 16)
        ),
        "assets/minecraft/textures/entity/chest/normal.png": _solid_png(
            (120, 80, 40, 255)
        ),
        "assets/minecraft/textures/entity/chest/normal_left.png": _solid_png(
            (120, 80, 41, 255)
        ),
        "assets/minecraft/textures/entity/chest/normal_right.png": _solid_png(
            (120, 80, 42, 255)
        ),
    }


def test_extract_emits_chest_quads(tmp_path):
    dest = tmp_path / "out.npz"
    summary = _extract_pack(
        _make_pack(tmp_path / "pack.zip", _chest_entries()), dest
    )
    # Two covered blocks: the chest and the texture-only planks fallback.
    assert summary.blocks == 2
    assert summary.skipped == 0
    data = _load_npz(dest)
    assert data["meta/version"].item() == "pack-quads.1"
    quads = data["quads/minecraft:chest[facing=north,type=single]"]
    assert len(quads) == 18
    assert set(str(k) for k in quads["texkey"]) == {"entity/chest/normal"}
    assert data["tex/entity/chest/normal"].shape == (64, 64, 4)
    expanded = [k for k in data.files if k.startswith("quads/minecraft:chest[")]
    assert len(expanded) == 12


def test_extract_chest_model_without_builtin_parent(tmp_path):
    entries = _chest_entries()
    entries["assets/minecraft/models/block/chest.json"] = json.dumps(
        {"textures": {"particle": "block/planks"}}
    )
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    assert summary.blocks == 2
    data = _load_npz(dest)
    # The expansion supersedes the variant stub: no planks-cube states.
    expanded = [k for k in data.files if k.startswith("quads/minecraft:chest[")]
    assert len(expanded) == 12
    assert (
        "quads/minecraft:chest[facing=north,type=single,waterlogged=false]"
        not in data.files
    )


def test_extract_partial_half_textures_skips_half_states(tmp_path):
    entries = _chest_entries()
    del entries["assets/minecraft/textures/entity/chest/normal_left.png"]
    del entries["assets/minecraft/textures/entity/chest/normal_right.png"]
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    assert summary.blocks == 2
    data = _load_npz(dest)
    expanded = sorted(
        k for k in data.files if k.startswith("quads/minecraft:chest[")
    )
    assert len(expanded) == 4
    assert all("type=single" in key for key in expanded)


def test_extract_without_entity_texture_leaves_chest_unresolved(tmp_path):
    entries = _chest_entries()
    del entries["assets/minecraft/textures/entity/chest/normal.png"]
    dest = tmp_path / "out.npz"
    summary = _extract_pack(_make_pack(tmp_path / "pack.zip", entries), dest)
    # planks is covered by the texture-only fallback; the chest is skipped.
    assert summary.blocks == 1
    assert summary.skipped == 1
    data = _load_npz(dest)
    assert not [
        key for key in data.files if key.startswith("quads/minecraft:chest")
    ]
    assert "tex/block/chest" in data.files


def test_chest_quads_default_cull_none():
    quads = chest_quads("minecraft:chest", "", _spy([]))
    assert (quads["cull"] == CULL_NONE).all()
