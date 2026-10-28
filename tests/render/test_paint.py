"""Painter contract: shading, magenta fallback, cutout, blending, order."""

import numpy as np
import pytest
from helpers.litematica_packer import build_litematic_bytes

from litematic import loads
from litematic.render.isometric import _compose
from litematic.render.pack import Textures
from litematic.render.painter import _MAGENTA, _SHADES, _paint
from litematic.render.quads import unit_cube_quads
from litematic.render.templates import (
    SHAPE_CUBE,
    SHAPE_SLAB_BOTTOM,
    _shape_templates,
)

# Temporary: the v2 quad painter replaces this module in plan Task 7.
pytestmark = pytest.mark.xfail(reason="v2 quad painter not wired yet (Task 7)")

_BG = (0, 0, 0, 0)
_STONE = (200, 30, 30, 255)
_GLASS = (160, 208, 224, 128)
_SLAB = (110, 84, 52, 255)


def _shaded(rgb, face):
    return tuple(round(c * _SHADES[face]) for c in rgb) + (255,)


def _textures(entries):
    arrays = {}
    quads_map = {}
    occl_map = {}
    for name, spec in entries.items():
        key = f"block/{name.split(':')[-1]}"
        if isinstance(spec, np.ndarray):
            arrays[key] = spec
            alpha = int(spec[..., 3].mean())
        else:
            arrays[key] = np.full((16, 16, 4), spec, np.uint8)
            alpha = spec[3]
        quads_map[f"{name}[]"] = unit_cube_quads(key)
        occl_map[f"{name}[]"] = alpha >= 250
    return Textures(
        version="pack-quads",
        arrays=arrays,
        quads_map=quads_map,
        occl_map=occl_map,
        pack_mtime=0.0,
    )


TEX = _textures(
    {
        "minecraft:stone": _STONE,
        "minecraft:glass": _GLASS,
        "minecraft:oak_slab": _SLAB,
    }
)


def _scene(blocks, textures=TEX):
    palette = [("minecraft:air", {})] + [
        (name, dict(props)) for name, props, *_ in blocks
    ]
    sx = max(b[2] for b in blocks) + 1
    sy = max(b[3] for b in blocks) + 1
    sz = max(b[4] for b in blocks) + 1
    idx = np.zeros(sx * sy * sz, np.uint16)
    for i, (_, _, x, y, z) in enumerate(blocks, start=1):
        idx[(y * sz + z) * sx + x] = i
    schem = loads(
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
    return _compose(schem, 0, textures)


def _gradient():
    image = np.zeros((16, 16, 4), np.uint8)
    for row in range(16):
        image[row, :, :3] = (row * 15, row * 15, row * 15)
    image[..., 3] = 255
    return image


def test_single_block_faces_shaded_per_face():
    scene = _scene([("minecraft:stone", {}, 0, 0, 0)])
    img = _paint(scene, 4, _BG)
    assert img.shape == (16, 16, 4)
    tm = _shape_templates(4)[SHAPE_CUBE]
    for pixels, face in ((tm.top, 0), (tm.east, 1), (tm.south, 2)):
        got = img[pixels[:, 0], pixels[:, 1]]
        assert (got == _shaded(_STONE[:3], face)).all(), face
    assert (img[..., 3] != 0).sum() == 12 * 16


def test_deep_scene_bottom_rows_stay_background():
    blocks = [
        ("minecraft:stone", {}, 0, 0, 0),
        ("minecraft:stone", {}, 1, 0, 1),
        ("minecraft:stone", {}, 0, 0, 2),
    ]
    img = _paint(_scene(blocks), 2, _BG)
    content_rows = np.nonzero(img[..., 3].any(axis=1))[0]
    assert content_rows.max() == img.shape[0] - 3
    assert (img[-2:, :, 3] == 0).all()


def test_pack_miss_renders_shaded_magenta():
    scene = _scene([("minecraft:gold_block", {}, 0, 0, 0)])
    img = _paint(scene, 1, _BG)
    tm = _shape_templates(1)[SHAPE_CUBE]
    for pixels, face in ((tm.top, 0), (tm.east, 1), (tm.south, 2)):
        got = img[pixels[:, 0], pixels[:, 1]]
        assert (got == _shaded(_MAGENTA[:3], face)).all(), face


def test_gradient_top_pixel_samples_low_texel_rows():
    tex = _textures({"minecraft:stone": _gradient()})
    scene = _scene([("minecraft:stone", {}, 0, 0, 0)], tex)
    img = _paint(scene, 4, _BG)
    tm = _shape_templates(4)[SHAPE_CUBE]
    first = tm.top[0]
    rows = tm.texel_top[:, 1]
    assert rows.max() - rows.min() >= 8  # sprite spans half the texture rows
    assert tm.texel_top[0][1] <= 1  # topmost sprite row samples a low texel row
    low = int(tm.texel_top[0][1])
    assert tuple(img[int(first[0]), int(first[1])]) == _shaded((low * 15,) * 3, 0)


def test_cutout_texels_not_written():
    texture = np.full((16, 16, 4), (90, 90, 200, 255), np.uint8)
    texture[:, ::2, 3] = 0
    tex = _textures({"minecraft:stone": texture})
    scene = _scene([("minecraft:stone", {}, 0, 0, 0)], tex)
    img = _paint(scene, 1, _BG)
    tm = _shape_templates(1)[SHAPE_CUBE]
    transparent = tm.texel_top[:, 1] % 2 == 0
    rows = tm.top[transparent, 0]
    cols = tm.top[transparent, 1]
    assert (img[rows, cols, 3] == 0).all()


def test_semi_transparent_alpha_preserved():
    tex = _textures({"minecraft:glass": _GLASS})
    scene = _scene([("minecraft:glass", {}, 0, 0, 0)], tex)
    img = _paint(scene, 1, _BG)
    assert (img[..., 3] == 128).sum() == 12


def test_glass_in_front_blends_over_stone():
    scene = _scene(
        [("minecraft:stone", {}, 0, 0, 0), ("minecraft:glass", {}, 1, 0, 0)]
    )
    img = _paint(scene, 1, _BG)
    stone_east = np.array(_shaded(_STONE[:3], 1)[:3], np.uint16)
    glass_south = np.array(_shaded(_GLASS[:3], 2)[:3], np.uint16)
    want = [
        round((glass_south[i] * 128 + stone_east[i] * 127) / 255) for i in range(3)
    ]
    assert tuple(img[2, 2]) == (*want, 255)


def test_glass_behind_stone_fully_covered():
    scene = _scene(
        [("minecraft:glass", {}, 0, 0, 0), ("minecraft:stone", {}, 1, 0, 0)]
    )
    img = _paint(scene, 1, _BG)
    assert tuple(img[2, 2]) == _shaded(_STONE[:3], 2)


def test_enclosed_pack_miss_paints_nothing():
    blocks = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if (x, y, z) == (1, 1, 1):
                    blocks.append(("minecraft:gold_block", {}, x, y, z))
                else:
                    blocks.append(("minecraft:stone", {}, x, y, z))
    img = _paint(_scene(blocks), 1, _BG)
    magenta = {
        _shaded(_MAGENTA[:3], 0),
        _shaded(_MAGENTA[:3], 1),
        _shaded(_MAGENTA[:3], 2),
    }
    flat = {
        tuple(px) for px in img.reshape(-1, 4).tolist() if px[3] != 0
    }
    assert not (flat & magenta)


def test_same_depth_pair_paints_both():
    scene = _scene(
        [("minecraft:stone", {}, 1, 0, 0), ("minecraft:stone", {}, 0, 1, 0)]
    )
    img = _paint(scene, 1, _BG)
    assert (img[..., 3] != 0).sum() == 24


def test_slab_bottom_paints_half_sprite():
    scene = _scene([("minecraft:oak_slab", {}, 0, 0, 0)])
    img = _paint(scene, 2, _BG)
    assert (img[..., 3] != 0).sum() == 8 * 4
    assert scene.shape.tolist() == [SHAPE_SLAB_BOTTOM]


def test_slab_top_differs_from_bottom():
    bottom = _paint(_scene([("minecraft:oak_slab", {}, 0, 0, 0)]), 2, _BG)
    top = _paint(
        _scene([("minecraft:oak_slab", {"type": "top"}, 0, 0, 0)]), 2, _BG
    )
    assert not np.array_equal(bottom, top)


def test_double_slab_paints_full_cube_sprite():
    scene = _scene([("minecraft:oak_slab", {"type": "double"}, 0, 0, 0)])
    img = _paint(scene, 2, _BG)
    assert (img[..., 3] != 0).sum() == 12 * 4


def test_background_fills_canvas():
    scene = _scene([("minecraft:stone", {}, 0, 0, 0)])
    img = _paint(scene, 1, (255, 255, 255, 255))
    assert (img[..., 3] == 255).all()


def test_paint_is_deterministic():
    scene = _scene([("minecraft:stone", {}, 0, 0, 0)])
    first = _paint(scene, 2, _BG)
    second = _paint(scene, 2, _BG)
    assert np.array_equal(first, second)


def test_flat_block_paints_top_rhombus_only():
    scene = _scene([("minecraft:rail", {}, 0, 0, 0)])
    img = _paint(scene, 2, _BG)
    opaque = np.nonzero(img[..., 3] > 0)
    assert len(opaque[0]) == 4 * 2 * 2
    assert opaque[0].max() < 2 * 2
