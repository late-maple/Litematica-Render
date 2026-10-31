"""Pack stack contract: override direction, mcmeta scoping, errors, cache."""

import io
import json
import os
import re
import zipfile

import pytest
from PIL import Image

from litematic.parse.model import BlockState
from litematic.render.errors import RenderError
from litematic.render.pack import load_pack
from litematic.render.pack import loader as pack_mod


def _png(color, size=(16, 16)):
    buf = io.BytesIO()
    Image.new("RGBA", size, (*color, 255)).save(buf, "PNG")
    return buf.getvalue()


def _make_pack(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


_FACES = ("down", "up", "north", "south", "west", "east")


def _base_pack(path):
    """Minimal geometry base: a full-cube stone with a 16px red texture."""
    return _make_pack(
        path,
        {
            "assets/minecraft/blockstates/stone.json": json.dumps(
                {"variants": {"": {"model": "block/stone"}}}
            ),
            "assets/minecraft/models/block/stone.json": json.dumps(
                {
                    "elements": [
                        {
                            "from": [0, 0, 0],
                            "to": [16, 16, 16],
                            "faces": {n: {"texture": "block/stone"} for n in _FACES},
                        }
                    ]
                }
            ),
            "assets/minecraft/textures/block/stone.png": _png((200, 30, 30)),
        },
    )


def test_texture_override_wins(tmp_path):
    base = _base_pack(tmp_path / "base.zip")
    addon = _make_pack(
        tmp_path / "addon.zip",
        {"assets/minecraft/textures/block/stone.png": _png((10, 20, 200), (32, 32))},
    )
    textures = load_pack(base, [addon])
    assert textures.arrays["block/stone"].shape == (32, 32, 4)
    assert (textures.arrays["block/stone"][0, 0] == (10, 20, 200, 255)).all()


def test_model_override_wins(tmp_path):
    base = _base_pack(tmp_path / "base.zip")
    addon = _make_pack(
        tmp_path / "addon.zip",
        {
            "assets/minecraft/models/block/stone.json": json.dumps(
                {
                    "elements": [
                        {
                            "from": [0, 0, 0],
                            "to": [16, 16, 16],
                            "faces": {n: {"texture": "block/blue"} for n in _FACES},
                        }
                    ]
                }
            ),
            "assets/minecraft/textures/block/blue.png": _png((10, 20, 200)),
        },
    )
    textures = load_pack(base, [addon])
    res = textures.resolve(BlockState("minecraft:stone"))
    assert res.miss is None
    assert {str(k) for k in res.quads["texkey"]} == {"block/blue"}


def test_blockstate_override_wins(tmp_path):
    base = _make_pack(
        tmp_path / "base.zip",
        {
            "assets/minecraft/blockstates/lamp.json": json.dumps(
                {"variants": {"": {"model": "block/a"}}}
            ),
            "assets/minecraft/models/block/a.json": json.dumps(
                {
                    "elements": [
                        {
                            "from": [0, 0, 0],
                            "to": [16, 16, 16],
                            "faces": {n: {"texture": "block/tex_a"} for n in _FACES},
                        }
                    ]
                }
            ),
            "assets/minecraft/textures/block/tex_a.png": _png((200, 30, 30)),
        },
    )
    addon = _make_pack(
        tmp_path / "addon.zip",
        {
            "assets/minecraft/blockstates/lamp.json": json.dumps(
                {
                    "variants": {
                        "": {"model": "block/a"},
                        "facing=south": {"model": "block/b"},
                    }
                }
            ),
            "assets/minecraft/models/block/b.json": json.dumps(
                {
                    "elements": [
                        {
                            "from": [0, 0, 0],
                            "to": [16, 16, 16],
                            "faces": {n: {"texture": "block/tex_b"} for n in _FACES},
                        }
                    ]
                }
            ),
            "assets/minecraft/textures/block/tex_b.png": _png((10, 20, 200)),
        },
    )
    textures = load_pack(base, [addon])
    south = textures.resolve(BlockState("minecraft:lamp", {"facing": "south"}))
    assert {str(k) for k in south.quads["texkey"]} == {"block/tex_b"}
    plain = textures.resolve(BlockState("minecraft:lamp"))
    assert {str(k) for k in plain.quads["texkey"]} == {"block/tex_a"}


def test_first_extra_wins(tmp_path):
    base = _base_pack(tmp_path / "base.zip")
    first = _make_pack(
        tmp_path / "first.zip",
        {"assets/minecraft/textures/block/stone.png": _png((1, 2, 3), (32, 32))},
    )
    second = _make_pack(
        tmp_path / "second.zip",
        {"assets/minecraft/textures/block/stone.png": _png((9, 9, 9), (48, 48))},
    )
    textures = load_pack(base, [first, second])
    assert textures.arrays["block/stone"].shape == (32, 32, 4)


# mcmeta scoping: metadata is searched from the stack top down to and
# including the pack that supplied the texture — never below it.


def test_override_texture_without_mcmeta_stays_static(tmp_path):
    base = _make_pack(
        tmp_path / "base.zip",
        {
            "assets/minecraft/blockstates/glowstone.json": json.dumps(
                {"variants": {"": {"model": "block/glowstone"}}}
            ),
            "assets/minecraft/models/block/glowstone.json": json.dumps(
                {
                    "elements": [
                        {
                            "from": [0, 0, 0],
                            "to": [16, 16, 16],
                            "faces": {
                                n: {"texture": "block/glowstone"} for n in _FACES
                            },
                        }
                    ]
                }
            ),
            "assets/minecraft/textures/block/glowstone.png": _png(
                (200, 200, 10), (16, 64)
            ),
            "assets/minecraft/textures/block/glowstone.png.mcmeta": json.dumps(
                {"animation": {}}
            ),
        },
    )
    addon = _make_pack(
        tmp_path / "addon.zip",
        {
            "assets/minecraft/textures/block/glowstone.png": _png(
                (10, 200, 10), (32, 96)
            ),
        },
    )
    textures = load_pack(base, [addon])
    # No inherited vanilla animation: the 3-frame strip stays whole.
    assert textures.arrays["block/glowstone"].shape == (96, 32, 4)


def test_override_texture_with_own_mcmeta_animates(tmp_path):
    base = _make_pack(
        tmp_path / "base.zip",
        {
            "assets/minecraft/blockstates/glowstone.json": json.dumps(
                {"variants": {"": {"model": "block/glowstone"}}}
            ),
            "assets/minecraft/models/block/glowstone.json": json.dumps(
                {
                    "elements": [
                        {
                            "from": [0, 0, 0],
                            "to": [16, 16, 16],
                            "faces": {
                                n: {"texture": "block/glowstone"} for n in _FACES
                            },
                        }
                    ]
                }
            ),
            "assets/minecraft/textures/block/glowstone.png": _png(
                (200, 200, 10), (16, 64)
            ),
            "assets/minecraft/textures/block/glowstone.png.mcmeta": json.dumps(
                {"animation": {}}
            ),
        },
    )
    addon = _make_pack(
        tmp_path / "addon.zip",
        {
            "assets/minecraft/textures/block/glowstone.png": _png(
                (10, 200, 10), (32, 96)
            ),
            "assets/minecraft/textures/block/glowstone.png.mcmeta": json.dumps(
                {"animation": {}}
            ),
        },
    )
    textures = load_pack(base, [addon])
    assert textures.arrays["block/glowstone"].shape == (32, 32, 4)


def test_mcmeta_only_extra_overrides_base_texture(tmp_path):
    base = _make_pack(
        tmp_path / "base.zip",
        {
            "assets/minecraft/blockstates/glowstone.json": json.dumps(
                {"variants": {"": {"model": "block/glowstone"}}}
            ),
            "assets/minecraft/models/block/glowstone.json": json.dumps(
                {
                    "elements": [
                        {
                            "from": [0, 0, 0],
                            "to": [16, 16, 16],
                            "faces": {
                                n: {"texture": "block/glowstone"} for n in _FACES
                            },
                        }
                    ]
                }
            ),
            "assets/minecraft/textures/block/glowstone.png": _png(
                (200, 200, 10), (16, 96)
            ),
        },
    )
    addon = _make_pack(
        tmp_path / "addon.zip",
        {
            "assets/minecraft/textures/block/glowstone.png.mcmeta": json.dumps(
                {"animation": {}}
            ),
        },
    )
    textures = load_pack(base, [addon])
    # The higher pack's mcmeta applies to the base texture: first frame only.
    assert textures.arrays["block/glowstone"].shape == (16, 16, 4)


# Add-on validation: a broken pack names itself and never passes silently.


def test_missing_extra_raises(tmp_path):
    base = _base_pack(tmp_path / "base.zip")
    with pytest.raises(RenderError, match="gone.zip"):
        load_pack(base, [tmp_path / "gone.zip"])


def test_direct_extract_missing_extra_names_error(tmp_path):
    """A vanished pack must surface as RenderError, not a raw OSError."""
    from litematic.render.pack import extract as extract_mod

    base = _base_pack(tmp_path / "base.zip")
    with pytest.raises(RenderError, match="gone.zip"):
        extract_mod._extract_pack(
            [base, tmp_path / "gone.zip"], tmp_path / "out.npz"
        )


def test_failed_stack_open_releases_packs(tmp_path):
    """Packs opened before a validation failure must not stay locked."""
    base = _base_pack(tmp_path / "base.zip")
    bare = _make_pack(tmp_path / "bare.zip", {"README.txt": "hi"})
    with pytest.raises(RenderError, match="no assets/ entries"):
        load_pack(base, [bare])
    os.remove(base)  # raises PermissionError on Windows if a handle leaked


def test_long_stem_stack_cache_name_bounded(tmp_path):
    """The stacked cache name stays a legal file name whatever the stems."""
    long = "l" * 120
    base = _base_pack(tmp_path / f"{long}_base.zip")
    addon = _make_pack(
        tmp_path / f"{long}_addon.zip",
        {"assets/minecraft/textures/block/stone.png": _png((10, 20, 200), (32, 32))},
    )
    textures = load_pack(base, [addon])
    (cache,) = tmp_path.glob("*.npz")
    assert len(cache.name) < 255
    assert textures.arrays["block/stone"].shape == (32, 32, 4)


def test_corrupt_extra_raises(tmp_path):
    base = _base_pack(tmp_path / "base.zip")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"definitely not a zip")
    with pytest.raises(RenderError, match="bad.zip"):
        load_pack(base, [bad])


def test_extra_without_assets_raises(tmp_path):
    base = _base_pack(tmp_path / "base.zip")
    bare = _make_pack(tmp_path / "bare.zip", {"README.txt": "hi"})
    with pytest.raises(RenderError, match="no assets/ entries"):
        load_pack(base, [bare])


def test_directory_extra_raises(tmp_path):
    base = _base_pack(tmp_path / "base.zip")
    folder = tmp_path / "folder.zip"
    folder.mkdir()
    with pytest.raises(RenderError, match="not a resource pack"):
        load_pack(base, [folder])


def test_extra_packs_type_error(tmp_path):
    base = _base_pack(tmp_path / "base.zip")
    with pytest.raises(RenderError, match="sequence of paths"):
        load_pack(base, 42)


def test_bare_string_extra_accepted(tmp_path):
    base = _base_pack(tmp_path / "base.zip")
    addon = _make_pack(
        tmp_path / "addon.zip",
        {"assets/minecraft/textures/block/stone.png": _png((10, 20, 200), (32, 32))},
    )
    textures = load_pack(base, str(addon))
    assert textures.arrays["block/stone"].shape == (32, 32, 4)


# Cache identity spans the whole stack; the single-pack path is untouched.


def test_stacked_cache_name_and_mtime_tuple(tmp_path):
    base = _base_pack(tmp_path / "base.zip")
    addon = _make_pack(
        tmp_path / "addon.zip",
        {"assets/minecraft/textures/block/stone.png": _png((10, 20, 200), (32, 32))},
    )
    textures = load_pack(base, [addon])
    caches = list(tmp_path.glob("*.npz"))
    assert len(caches) == 1
    assert re.fullmatch(r"base\+addon\.[0-9a-f]{8}\.litematic-tex\.npz", caches[0].name)
    assert isinstance(textures.pack_mtime, tuple)
    assert len(textures.pack_mtime) == 2


def test_stacked_second_load_hits_disk_cache(tmp_path, monkeypatch):
    base = _base_pack(tmp_path / "base.zip")
    addon = _make_pack(
        tmp_path / "addon.zip",
        {"assets/minecraft/textures/block/stone.png": _png((10, 20, 200), (32, 32))},
    )
    load_pack(base, [addon])
    monkeypatch.setattr(pack_mod, "_PACK_CACHE", {})
    counter = pack_mod._extract_pack
    calls = []

    def counting(sources, dest, pack_mtime=None):
        calls.append(dest)
        return counter(sources, dest, pack_mtime=pack_mtime)

    monkeypatch.setattr(pack_mod, "_extract_pack", counting)
    textures = load_pack(base, [addon])
    assert calls == []
    assert textures.arrays["block/stone"].shape == (32, 32, 4)


def test_extra_mtime_bump_forces_reextract(tmp_path, monkeypatch):
    base = _base_pack(tmp_path / "base.zip")
    addon = _make_pack(
        tmp_path / "addon.zip",
        {"assets/minecraft/textures/block/stone.png": _png((10, 20, 200), (32, 32))},
    )
    load_pack(base, [addon])
    real = pack_mod._extract_pack
    calls = []

    def counting(sources, dest, pack_mtime=None):
        calls.append(dest)
        return real(sources, dest, pack_mtime=pack_mtime)

    monkeypatch.setattr(pack_mod, "_extract_pack", counting)
    stat = addon.stat()
    os.utime(addon, (stat.st_atime + 10, stat.st_mtime + 10))
    textures = load_pack(base, [addon])
    assert len(calls) == 1
    assert len(textures.pack_mtime) == 2


def test_stacked_load_leaves_legacy_cache_alone(tmp_path):
    base = _base_pack(tmp_path / "base.zip")
    addon = _make_pack(
        tmp_path / "addon.zip",
        {"assets/minecraft/textures/block/stone.png": _png((10, 20, 200), (32, 32))},
    )
    load_pack(base, [addon])
    assert not (tmp_path / "base.litematic-tex.npz").exists()
