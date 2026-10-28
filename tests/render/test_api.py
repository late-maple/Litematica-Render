"""Public API contract: params, errors, PNG roundtrip, unresolved names."""

import io
import json
import zipfile

import numpy as np
import pytest
from helpers.litematica_packer import build_litematic_bytes
from PIL import Image

import litematic
from litematic import loads
from litematic.render import (
    RenderError,
    isometric,
    isometric_png,
    to_png,
    unresolved_colors,
)

_FACES = ("down", "up", "north", "south", "west", "east")


def _png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGBA", (16, 16), (*color, 255)).save(buf, "PNG")
    return buf.getvalue()


def _two_color_pack(path):
    entries = {}
    for name, color in (("stone", (255, 0, 0)), ("gold_block", (0, 0, 255))):
        entries[f"assets/minecraft/blockstates/{name}.json"] = json.dumps(
            {"variants": {"": {"model": f"block/{name}"}}}
        )
        faces = {face: {"texture": f"block/{name}"} for face in _FACES}
        entries[f"assets/minecraft/models/block/{name}.json"] = json.dumps(
            {"elements": [{"from": [0, 0, 0], "to": [16, 16, 16], "faces": faces}]}
        )
        entries[f"assets/minecraft/textures/block/{name}.png"] = _png_bytes(color)
    with zipfile.ZipFile(path, "w") as zf:
        for key, data in entries.items():
            zf.writestr(key, data)
    return path


def _schem(blocks):
    names = [n if ":" in n else f"minecraft:{n}" for n, *_ in blocks]
    palette = [("minecraft:air", {})] + [(name, {}) for name in names]
    sx = max(b[1] for b in blocks) + 1
    sy = max(b[2] for b in blocks) + 1
    sz = max(b[3] for b in blocks) + 1
    idx = np.zeros(sx * sy * sz, np.uint16)
    for i, (_, x, y, z) in enumerate(blocks, start=1):
        idx[(y * sz + z) * sx + x] = i
    return loads(
        build_litematic_bytes(
            [
                {
                    "name": "Main",
                    "position": (0, 0, 0),
                    "size": (sx, sy, sz),
                    "palette": palette,
                    "indices_yzx": idx,
                }
            ]
        )
    )


_TWO = [("stone", 0, 0, 0), ("gold_block", 1, 0, 0)]


def test_all_exports_have_docstrings():
    import litematic.render as mod

    for name in mod.__all__:
        assert getattr(mod, name).__doc__, name


def test_top_level_all_unchanged():
    assert litematic.__all__ == [
        "BlockEntity",
        "BlockState",
        "Entity",
        "Litematic",
        "LitematicError",
        "LitematicFormatError",
        "LitematicMetadata",
        "NbtError",
        "Region",
        "Vec3i",
        "load",
        "loads",
    ]


def test_textures_path_required():
    with pytest.raises(RenderError, match="resource pack"):
        isometric(_schem(_TWO))


@pytest.mark.xfail(reason="v2 quad pipeline not wired yet (Task 6/8)")
def test_all_air_rejected(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = loads(
        build_litematic_bytes(
            [
                {
                    "name": "Main",
                    "position": (0, 0, 0),
                    "size": (1, 1, 1),
                    "palette": [("minecraft:air", {})],
                    "indices_yzx": np.zeros(1, np.uint16),
                }
            ]
        )
    )
    with pytest.raises(RenderError, match="no blocks to render"):
        isometric(schem, textures_path=pack)


def test_yaw_out_of_range():
    with pytest.raises(ValueError):
        isometric(_schem(_TWO), textures_path=None, yaw=4)


def test_param_bounds_rejected(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    for kwargs in (
        {"max_size": 9000},
        {"max_size": 15},
        {"supersample": 0},
        {"supersample": 5},
        {"background": (256, 0, 0, 0)},
        {"background": (0, 0, 0)},
        {"background": (0, 0, 0, 0.5)},
    ):
        with pytest.raises(RenderError):
            isometric(schem, textures_path=pack, **kwargs)


def test_missing_pack_path(tmp_path):
    with pytest.raises(RenderError):
        isometric(_schem(_TWO), textures_path=tmp_path / "missing.zip")


@pytest.mark.xfail(reason="v2 quad pipeline not wired yet (Task 7/8)")
def test_png_decodes_to_array_and_shows_both_blocks(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    png = isometric_png(schem, textures_path=pack, max_size=64)
    arr = isometric(schem, textures_path=pack, max_size=64)
    with Image.open(io.BytesIO(png)) as im:
        decoded = np.array(im.convert("RGBA"))
    assert (decoded == arr).all()
    assert arr.shape[0] <= 64 and arr.shape[1] <= 64
    colors = {tuple(c) for c in arr.reshape(-1, 4)}
    assert any(c[0] > c[2] and c[3] == 255 for c in colors)
    assert any(c[2] > c[0] and c[3] == 255 for c in colors)


@pytest.mark.xfail(reason="v2 quad pipeline not wired yet (Task 7/8)")
def test_isometric_png_deterministic(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    first = isometric_png(schem, textures_path=pack, max_size=64)
    second = isometric_png(schem, textures_path=pack, max_size=64)
    assert first == second


def test_unresolved_colors(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem([("stone", 0, 0, 0), ("create:x", 1, 0, 0)])
    assert unresolved_colors(schem, textures_path=pack) == {"create:x"}


def test_to_png_signature_and_rejection():
    img = np.full((1, 1, 4), (1, 2, 3, 255), np.uint8)
    assert to_png(img).startswith(b"\x89PNG")
    with pytest.raises(RenderError):
        to_png(np.zeros((2, 2, 3), np.uint8))
