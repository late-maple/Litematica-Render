"""Pack loading contract: extraction cache, mtime keying, state-aware faces."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from litematic.render.errors import RenderError
from litematic.render.pack import Textures, load_pack
from litematic.render.pack import loader as pack_mod
from litematic.render.quads import TIER_EAST, TIER_SOUTH, TIER_UP, unit_cube_quads


def _png(color, size=(16, 16)):
    buf = io.BytesIO()
    Image.new("RGBA", size, (*color, 255)).save(buf, "PNG")
    return buf.getvalue()


def _mini_pack(path, color=(200, 30, 30)):
    all_faces = {n: {"texture": "block/stone"} for n in _FACE_NAMES}
    entries = {
        "assets/minecraft/blockstates/stone.json": json.dumps(
            {"variants": {"": {"model": "block/stone"}}}
        ),
        "assets/minecraft/models/block/stone.json": json.dumps(
            {
                "elements": [
                    {"from": [0, 0, 0], "to": [16, 16, 16], "faces": all_faces}
                ]
            }
        ),
        "assets/minecraft/textures/block/stone.png": _png(color),
    }
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


class _Counter:
    def __init__(self):
        self.calls = 0
        self.real = pack_mod._extract_pack

    def __call__(self, source, dest, pack_mtime=None):
        self.calls += 1
        return self.real(source, dest, pack_mtime=pack_mtime)


_FACE_NAMES = ("down", "up", "north", "south", "west", "east")


def test_first_load_extracts_and_caches(tmp_path):
    pack_file = _mini_pack(tmp_path / "pack.zip")
    textures = load_pack(pack_file)
    assert isinstance(textures, Textures)
    assert textures.version == "pack-quads.5"
    assert (tmp_path / "pack.litematic-tex.npz").is_file()
    quads = textures.quads("minecraft:stone", {})
    assert quads is not None
    assert len(quads) == 6
    assert textures.occludes("minecraft:stone", {})


def test_second_load_reuses_cache(tmp_path):
    pack_file = _mini_pack(tmp_path / "pack.zip")
    counter = _Counter()
    load_pack(pack_file)
    pack_mod._extract_pack = counter
    try:
        textures = load_pack(pack_file)
    finally:
        pack_mod._extract_pack = counter.real
    assert counter.calls == 0
    assert textures.quads("minecraft:stone", {}) is not None


def test_pack_mtime_change_forces_reextract(tmp_path):
    pack_file = _mini_pack(tmp_path / "pack.zip")
    counter = _Counter()
    pack_mod._extract_pack = counter
    try:
        load_pack(pack_file)
        _mini_pack(pack_file, color=(10, 200, 10))
        textures = load_pack(pack_file)
    finally:
        pack_mod._extract_pack = counter.real
    assert counter.calls == 2
    quads = textures.quads("minecraft:stone", {})
    assert quads is not None
    key = str(quads["texkey"][0])
    assert (textures.arrays[key] == (10, 200, 10, 255)).all()


def test_corrupt_cache_rebuilt(tmp_path):
    pack_file = _mini_pack(tmp_path / "pack.zip")
    cache = tmp_path / "pack.litematic-tex.npz"
    cache.write_bytes(b"garbage not npz")
    counter = _Counter()
    pack_mod._extract_pack = counter
    try:
        textures = load_pack(pack_file)
    finally:
        pack_mod._extract_pack = counter.real
    assert counter.calls == 1
    assert textures.quads("minecraft:stone", {}) is not None


def test_cache_missing_meta_version_rebuilt(tmp_path):
    pack_file = _mini_pack(tmp_path / "pack.zip")
    cache = tmp_path / "pack.litematic-tex.npz"
    np.savez(cache, tex__block_stone=np.zeros((16, 16, 4), np.uint8))
    counter = _Counter()
    pack_mod._extract_pack = counter
    try:
        load_pack(pack_file)
    finally:
        pack_mod._extract_pack = counter.real
    assert counter.calls == 1


def test_bad_shape_entry_dropped(tmp_path):
    pack_file = _mini_pack(tmp_path / "pack.zip")
    load_pack(pack_file)
    cache = tmp_path / "pack.litematic-tex.npz"
    data = dict(np.load(cache, allow_pickle=False))
    data["quads/minecraft:stone[]"] = np.zeros((3, 4), np.uint8)
    np.savez(cache, **data)
    counter = _Counter()
    pack_mod._extract_pack = counter
    try:
        textures = load_pack(pack_file)
    finally:
        pack_mod._extract_pack = counter.real
    assert "minecraft:stone[]" not in textures.quads_map
    assert textures.quads("minecraft:stone", {}) is None


def test_quads_entry_with_unknown_texkey_dropped(tmp_path):
    pack_file = _mini_pack(tmp_path / "pack.zip")
    load_pack(pack_file)
    cache = tmp_path / "pack.litematic-tex.npz"
    data = dict(np.load(cache, allow_pickle=False))
    data["quads/minecraft:ghost[]"] = unit_cube_quads("block/ghost")
    np.savez(cache, **data)
    textures = load_pack(pack_file)
    assert textures.quads("minecraft:ghost", {}) is None
    assert textures.quads("minecraft:stone", {}) is not None


def test_state_lookup_with_fallback(tmp_path):
    faces = {n: {"texture": f"block/{n}"} for n in ("up", "east", "south")}
    entries = {
        "assets/minecraft/blockstates/thing.json": json.dumps(
            {
                "variants": {
                    "facing=south": {"model": "block/thing"},
                    "facing=north": {"model": "block/thing", "y": 180},
                }
            }
        ),
        "assets/minecraft/models/block/thing.json": json.dumps(
            {"elements": [{"from": [0] * 3, "to": [16] * 3, "faces": faces}]}
        ),
        "assets/minecraft/textures/block/up.png": _png((1, 2, 3)),
        "assets/minecraft/textures/block/east.png": _png((4, 5, 6)),
        "assets/minecraft/textures/block/south.png": _png((7, 8, 9)),
        "assets/minecraft/blockstates/plain.json": json.dumps(
            {"variants": {"": {"model": "block/thing"}}}
        ),
    }
    with zipfile.ZipFile(tmp_path / "p2.zip", "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    textures = load_pack(tmp_path / "p2.zip")
    quads = textures.quads("minecraft:thing", {"facing": "south"})
    assert quads is not None
    by_tier = {int(q["shade"]): str(q["texkey"]) for q in quads}
    assert by_tier[TIER_UP] == "block/up"
    assert by_tier[TIER_EAST] == "block/east"
    assert by_tier[TIER_SOUTH] == "block/south"
    assert (textures.arrays["block/up"] == (1, 2, 3, 255)).all()
    assert textures.quads("minecraft:thing", {"facing": "west"}) is None
    assert textures.quads("minecraft:plain", {"anything": "x"}) is not None
    assert textures.quads("minecraft:absent", {}) is None
    assert not textures.occludes("minecraft:thing", {"facing": "south"})
    assert not textures.occludes("minecraft:plain", {})
    assert not textures.occludes("minecraft:absent", {})


def test_missing_pack_raises(tmp_path):
    with pytest.raises(RenderError, match="load:"):
        load_pack(tmp_path / "nope.zip")


def test_corrupt_cache_member_auto_rebuilds(tmp_path):
    pack_file = _mini_pack(tmp_path / "pack.zip")
    load_pack(pack_file)
    cache = tmp_path / "pack.litematic-tex.npz"
    raw = bytearray(cache.read_bytes())
    member = raw.index(b"tex/block/stone")
    raw[member + 120] ^= 0xFF  # flip a byte inside the compressed payload
    cache.write_bytes(bytes(raw))
    counter = _Counter()
    pack_mod._extract_pack = counter
    try:
        textures = load_pack(pack_file)
    finally:
        pack_mod._extract_pack = counter.real
    assert counter.calls == 1
    assert textures.quads("minecraft:stone", {}) is not None


def test_unwritable_cache_falls_back_to_tempdir(tmp_path, monkeypatch):
    from litematic.render.pack import extract as extract_mod

    pack_file = _mini_pack(tmp_path / "pack.zip")
    real = pack_mod._extract_pack
    calls = []

    def flaky(source, dest, pack_mtime=None):
        calls.append(Path(dest))
        if len(calls) == 1:
            raise extract_mod._CacheWriteError("extract: cannot write npz: boom")
        return real(source, dest, pack_mtime=pack_mtime)

    monkeypatch.setattr(pack_mod, "_extract_pack", flaky)
    textures = load_pack(pack_file)
    assert len(calls) == 2
    assert calls[0].name == "pack.litematic-tex.npz"
    assert calls[1].parent != calls[0].parent
    assert textures.quads("minecraft:stone", {}) is not None


def test_vanished_pack_raises_render_error(tmp_path, monkeypatch):
    from pathlib import Path as PathType

    monkeypatch.setattr(PathType, "is_file", lambda self: True)

    def boom(self):
        raise FileNotFoundError(self)

    monkeypatch.setattr(PathType, "stat", boom)
    with pytest.raises(RenderError, match="load:"):
        load_pack(tmp_path / "gone.zip")


def test_two_arg_extract_produces_loadable_cache(tmp_path):
    from litematic.render.pack import extract as extract_mod

    pack_file = _mini_pack(tmp_path / "pack.zip")
    dest = tmp_path / "out.npz"
    extract_mod._extract_pack(pack_file, dest)
    parsed = pack_mod._parse_textures(dest)
    assert parsed is not None
    assert parsed.version == "pack-quads.5"


def test_old_cache_version_rebuilds(tmp_path):
    pack_file = _mini_pack(tmp_path / "pack.zip")
    load_pack(pack_file)
    cache = tmp_path / "pack.litematic-tex.npz"
    data = dict(np.load(cache, allow_pickle=False))
    data["meta/version"] = np.array("pack")  # simulate a v1 cache
    np.savez(cache, **data)
    counter = _Counter()
    pack_mod._extract_pack = counter
    try:
        textures = load_pack(pack_file)
    finally:
        pack_mod._extract_pack = counter.real
    assert counter.calls == 1
    assert textures.version == "pack-quads.5"
    assert textures.quads("minecraft:stone", {}) is not None


def test_load_pack_missing_pillow_still_render_error(tmp_path, monkeypatch):
    import sys

    pack_file = _mini_pack(tmp_path / "pack.zip")
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(RenderError, match="Pillow is required"):
        load_pack(pack_file)
    assert not (tmp_path / "pack.litematic-tex.npz").exists()


# Subset state matching: npz keys are the pack's own (partial) variant
# keys, so a schematic state with extra properties (waterlogged=false and
# friends) must still match the most specific key it satisfies.


def _partial_key_pack(path):
    """Pack with two overlapping partial variant keys, distinct textures."""
    faces_a = {"south": {"texture": "block/tex_a"}}
    faces_b = {"south": {"texture": "block/tex_b"}}
    return _make_pack(
        path,
        {
            "assets/minecraft/blockstates/thing.json": json.dumps(
                {
                    "variants": {
                        "facing=north": {"model": "block/a"},
                        "facing=north,half=bottom": {"model": "block/b"},
                    }
                }
            ),
            "assets/minecraft/models/block/a.json": json.dumps(
                {"elements": [{"from": [0] * 3, "to": [16] * 3, "faces": faces_a}]}
            ),
            "assets/minecraft/models/block/b.json": json.dumps(
                {"elements": [{"from": [0] * 3, "to": [16] * 3, "faces": faces_b}]}
            ),
            "assets/minecraft/textures/block/tex_a.png": _png((1, 2, 3)),
            "assets/minecraft/textures/block/tex_b.png": _png((4, 5, 6)),
        },
    )


def _make_pack(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def test_quads_match_ignores_extra_state_properties(tmp_path):
    pack = _mini_pack(tmp_path / "p.zip")
    entries = {
        "assets/minecraft/blockstates/thing.json": json.dumps(
            {"variants": {"facing=north": {"model": "block/thing"}}}
        ),
        "assets/minecraft/models/block/thing.json": json.dumps(
            {
                "elements": [
                    {
                        "from": [0] * 3,
                        "to": [16] * 3,
                        "faces": {n: {"texture": "block/stone"} for n in _FACE_NAMES},
                    }
                ]
            }
        ),
        "assets/minecraft/textures/block/stone.png": _png((1, 2, 3)),
    }
    pack = _make_pack(pack, entries)
    textures = load_pack(pack)
    quads = textures.quads(
        "minecraft:thing", {"facing": "north", "waterlogged": "false"}
    )
    assert quads is not None
    assert len(quads) == 6


def test_quads_most_specific_key_wins(tmp_path):
    textures = load_pack(_partial_key_pack(tmp_path / "p.zip"))
    quads = textures.quads(
        "minecraft:thing",
        {"facing": "north", "half": "bottom", "waterlogged": "false"},
    )
    assert quads is not None
    assert set(str(k) for k in quads["texkey"]) == {"block/tex_b"}
    loose = textures.quads("minecraft:thing", {"facing": "north"})
    assert loose is not None
    assert set(str(k) for k in loose["texkey"]) == {"block/tex_a"}


def test_quads_conflicting_value_not_matched(tmp_path):
    entries = {
        "assets/minecraft/blockstates/thing.json": json.dumps(
            {"variants": {"waterlogged=true": {"model": "block/thing"}}}
        ),
        "assets/minecraft/models/block/thing.json": json.dumps(
            {
                "elements": [
                    {
                        "from": [0] * 3,
                        "to": [16] * 3,
                        "faces": {n: {"texture": "block/stone"} for n in _FACE_NAMES},
                    }
                ]
            }
        ),
        "assets/minecraft/textures/block/stone.png": _png((1, 2, 3)),
    }
    textures = load_pack(_make_pack(tmp_path / "p.zip", entries))
    assert (
        textures.quads("minecraft:thing", {"waterlogged": "false"}) is None
    )
    assert (
        textures.quads("minecraft:thing", {"waterlogged": "true"}) is not None
    )


def test_occludes_match_ignores_extra_state_properties(tmp_path):
    pack = _mini_pack(tmp_path / "p.zip")
    entries = {
        "assets/minecraft/blockstates/thing.json": json.dumps(
            {"variants": {"facing=north": {"model": "block/thing"}}}
        ),
        "assets/minecraft/models/block/thing.json": json.dumps(
            {
                "elements": [
                    {
                        "from": [0] * 3,
                        "to": [16] * 3,
                        "faces": {n: {"texture": "block/stone"} for n in _FACE_NAMES},
                    }
                ]
            }
        ),
        "assets/minecraft/textures/block/stone.png": _png((1, 2, 3)),
    }
    textures = load_pack(_make_pack(pack, entries))
    assert textures.occludes(
        "minecraft:thing", {"facing": "north", "waterlogged": "false"}
    )


def test_multipart_malformed_when_matches_nothing(tmp_path):
    faces = {n: {"texture": "block/tex"} for n in _FACE_NAMES}
    entries = {
        "assets/minecraft/blockstates/thing.json": json.dumps(
            {
                "multipart": [
                    {
                        "when": {"OR": ["facing=north"]},
                        "apply": {"model": "block/bad"},
                    },
                    {"apply": {"model": "block/plain"}},
                ]
            }
        ),
        "assets/minecraft/models/block/bad.json": json.dumps(
            {"elements": [{"from": [0] * 3, "to": [16] * 3, "faces": faces}]}
        ),
        "assets/minecraft/models/block/plain.json": json.dumps(
            {"elements": [{"from": [0] * 3, "to": [16] * 3, "faces": faces}]}
        ),
        "assets/minecraft/textures/block/tex.png": _png((1, 2, 3)),
    }
    textures = load_pack(_make_pack(tmp_path / "p.zip", entries))
    quads = textures.quads("minecraft:thing", {"facing": "north"})
    assert quads is not None
    assert len(quads) == 6


def test_native_shape_cache_accepted(tmp_path):
    """tex/ arrays of any (h, w, 4) uint8 shape load through unchanged."""
    pack_file = _mini_pack(tmp_path / "pack.zip")
    load_pack(pack_file)
    cache = tmp_path / "pack.litematic-tex.npz"
    data = dict(np.load(cache, allow_pickle=False))
    data["tex/block/wide"] = np.zeros((8, 32, 4), np.uint8)
    np.savez(cache, **data)
    textures = load_pack(pack_file)
    assert textures.arrays["block/wide"].shape == (8, 32, 4)


def test_native_texture_pack_renders_quads(tmp_path):
    """A 32x32 pack texture flows through load_pack into usable quads."""
    faces = {n: {"texture": "block/stone"} for n in _FACE_NAMES}
    entries = {
        "assets/minecraft/blockstates/stone.json": json.dumps(
            {"variants": {"": {"model": "block/stone"}}}
        ),
        "assets/minecraft/models/block/stone.json": json.dumps(
            {"elements": [{"from": [0, 0, 0], "to": [16, 16, 16], "faces": faces}]}
        ),
        "assets/minecraft/textures/block/stone.png": _png((1, 2, 3), size=(32, 32)),
    }
    with zipfile.ZipFile(tmp_path / "p.zip", "w") as zf:
        for name, blob in entries.items():
            zf.writestr(name, blob)
    textures = load_pack(tmp_path / "p.zip")
    assert textures.arrays["block/stone"].shape == (32, 32, 4)
    assert textures.quads("minecraft:stone", {}) is not None
