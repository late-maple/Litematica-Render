"""Shared render-test fixtures: synthetic resource pack + schematics."""

import io
import json
import zipfile

import numpy as np

from helpers.litematica_packer import build_litematic_bytes
from litematic import loads

_FACES = ("down", "up", "north", "south", "west", "east")

_TWO = [("stone", 0, 0, 0), ("gold_block", 1, 0, 0)]


def _png_bytes(color):
    from PIL import Image

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


def _schem(blocks, entities=None):
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
                    "entities": entities or [],
                }
            ]
        )
    )
