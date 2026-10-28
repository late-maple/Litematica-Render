"""Pack loading contract: extraction cache, mtime keying, state-aware faces."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from litematic.render import pack as pack_mod
from litematic.render.errors import RenderError
from litematic.render.pack import Textures, load_pack


def _png(color):
    buf = io.BytesIO()
    Image.new("RGBA", (16, 16), (*color, 255)).save(buf, "PNG")
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
    assert textures.version == "pack"
    assert (tmp_path / "pack.litematic-tex.npz").is_file()
    up, east, south = textures.faces("minecraft:stone", {})
    assert (up == (200, 30, 30, 255)).all()
    assert east is not None and south is not None


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
    assert textures.faces("minecraft:stone", {}) is not None


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
    up, _, _ = textures.faces("minecraft:stone", {})
    assert (up == (10, 200, 10, 255)).all()


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
    assert textures.faces("minecraft:stone", {}) is not None


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
    data["tex/block/stone"] = np.zeros((8, 8, 4), np.uint8)
    np.savez(cache, **data)
    counter = _Counter()
    pack_mod._extract_pack = counter
    try:
        textures = load_pack(pack_file)
    finally:
        pack_mod._extract_pack = counter.real
    assert "block/stone" not in textures.arrays
    assert textures.faces("minecraft:stone", {}) is None


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
    up, east, south = textures.faces("minecraft:thing", {"facing": "south"})
    assert (up == (1, 2, 3, 255)).all()
    assert (east == (4, 5, 6, 255)).all()
    unknown = textures.faces("minecraft:thing", {"facing": "west"})
    assert unknown is None
    assert textures.faces("minecraft:plain", {"anything": "x"}) is not None
    assert textures.faces("minecraft:absent", {}) is None


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
    assert textures.faces("minecraft:stone", {}) is not None


def test_unwritable_cache_falls_back_to_tempdir(tmp_path, monkeypatch):
    from litematic.render import extract as extract_mod

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
    assert textures.faces("minecraft:stone", {}) is not None


def test_vanished_pack_raises_render_error(tmp_path, monkeypatch):
    from pathlib import Path as PathType

    monkeypatch.setattr(PathType, "is_file", lambda self: True)

    def boom(self):
        raise FileNotFoundError(self)

    monkeypatch.setattr(PathType, "stat", boom)
    with pytest.raises(RenderError, match="load:"):
        load_pack(tmp_path / "gone.zip")


def test_two_arg_extract_produces_loadable_cache(tmp_path):
    from litematic.render import extract as extract_mod

    pack_file = _mini_pack(tmp_path / "pack.zip")
    dest = tmp_path / "out.npz"
    extract_mod._extract_pack(pack_file, dest)
    parsed = pack_mod._parse_textures(dest)
    assert parsed is not None
    assert parsed.version == "pack"
