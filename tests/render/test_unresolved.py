"""unresolved_colors contract: state-aware magenta resolution."""

import io
import json
import zipfile

import numpy as np
from helpers.litematica_packer import build_litematic_bytes
from PIL import Image

from litematic import loads
from litematic.render import unresolved_colors

_FACES = ("down", "up", "north", "south", "west", "east")


def _png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGBA", (16, 16), (*color, 255)).save(buf, "PNG")
    return buf.getvalue()


def _write_pack(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for key, data in entries.items():
            zf.writestr(key, data)
    return path


def _cube_model(texture):
    faces = {face: {"texture": texture} for face in _FACES}
    return json.dumps(
        {"elements": [{"from": [0, 0, 0], "to": [16, 16, 16], "faces": faces}]}
    )


def _state_key_pack(path):
    """Pack whose zorp cube exists only under the facing=north state key."""
    return _write_pack(
        path,
        {
            "assets/minecraft/blockstates/zorp.json": json.dumps(
                {"variants": {"facing=north": {"model": "block/zorp"}}}
            ),
            "assets/minecraft/models/block/zorp.json": _cube_model(
                "block/zorp"
            ),
            "assets/minecraft/textures/block/zorp.png": _png_bytes(
                (10, 20, 30)
            ),
        },
    )


def _texture_only_pack(path, stem):
    """Pack with one block texture and no blockstates at all."""
    return _write_pack(
        path,
        {f"assets/minecraft/textures/block/{stem}.png": _png_bytes((40, 50, 60))},
    )


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


def test_state_key_entry_leaves_other_states_unresolved(tmp_path):
    pack = _state_key_pack(tmp_path / "pack.zip")
    schem = _schem([("zorp", 0, 0, 0)])
    assert unresolved_colors(schem, textures_path=pack) == {"minecraft:zorp"}


def test_texture_named_for_block_is_covered(tmp_path):
    pack = _texture_only_pack(tmp_path / "pack.zip", "oak_stairs")
    schem = _schem([("oak_stairs", 0, 0, 0)])
    assert unresolved_colors(schem, textures_path=pack) == set()


def test_name_without_pack_entry_is_unresolved(tmp_path):
    pack = _state_key_pack(tmp_path / "pack.zip")
    schem = _schem([("create:x", 0, 0, 0)])
    assert unresolved_colors(schem, textures_path=pack) == {"create:x"}
