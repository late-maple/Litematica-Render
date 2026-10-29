"""Painter v3 contract: quad rasterization, vanilla lighting, alpha bands."""

import numpy as np
import pytest
from helpers.litematica_packer import build_litematic_bytes

from litematic import loads
from litematic.render.errors import RenderError
from litematic.render.pack import Textures
from litematic.render.quads import QUAD_DTYPE, TIER_SOUTH, unit_cube_quads
from litematic.render.scene.isometric import _compose
from litematic.render.scene.painter import _paint

_BG = (0, 0, 0, 0)
_STONE = (200, 30, 30, 255)
_GLASS = (160, 208, 224, 128)
_PLANT = (60, 180, 80, 255)
_ODD = (90, 200, 90, 255)


def _textures(entries):
    """Pack with one unit-cube state per entry, occluding when opaque."""
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
        states=None,
        parts_map={},
        mquad_map={},
    )


TEX = _textures({"minecraft:stone": _STONE, "minecraft:glass": _GLASS})


def _raw_quad_pack(quads, color):
    """Pack exposing one raw quad array backed by a uniform texture."""
    key = str(quads["texkey"][0])
    return Textures(
        version="pack-quads",
        arrays={key: np.full((16, 16, 4), color, np.uint8)},
        quads_map={"minecraft:odd[]": quads},
        occl_map={"minecraft:odd[]": False},
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map={},
    )


def _scene(blocks, textures=TEX):
    """Compose the given (name, properties, x, y, z) blocks into a scene."""
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


def test_isolated_cube_face_brightness_vanilla():
    img = _paint(_scene([("minecraft:stone", {}, 0, 0, 0)]), 4, _BG)
    assert img.shape == (16, 16, 4)
    assert tuple(img[4, 8]) == (200, 30, 30, 255)  # top, light 1.0
    assert tuple(img[10, 4]) == (160, 24, 24, 255)  # south, light 0.8
    assert tuple(img[10, 12]) == (120, 18, 18, 255)  # east, light 0.6
    painted = {tuple(px) for px in img.reshape(-1, 4).tolist() if px[3]}
    assert painted == {
        (200, 30, 30, 255),
        (160, 24, 24, 255),
        (120, 18, 18, 255),
    }
    assert int((img[..., 3] != 0).sum()) == 192


def test_ao_darkens_wall_base_pixels():
    cells = [
        ("minecraft:stone", {}, x, 0, z) for x in range(3) for z in range(3)
    ]
    cells += [("minecraft:stone", {}, x, 1, 0) for x in range(3)]
    img = _paint(_scene(cells), 4, _BG)
    assert tuple(img[8, 32]) == (200, 30, 30, 255)  # wall top, ao = 1
    assert img[18, 26, 0] < 160  # wall-adjacent top-face pixel darkened
    assert img[18, 26, 0] < img[20, 24, 0] < 200


def test_cross_plant_renders_double_sided():
    arrays = {"block/short_grass": np.full((16, 16, 4), _PLANT, np.uint8)}
    tex = Textures(
        version="pack-quads",
        arrays=arrays,
        quads_map={},
        occl_map={},
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map={},
    )
    scene = _scene([("minecraft:short_grass", {}, 0, 0, 0)], tex)
    assert scene.quad_dbl.all()
    img = _paint(scene, 4, _BG)
    assert tuple(img[8, 8]) == _PLANT
    painted = np.nonzero(img[..., 3])
    assert len(painted[0]) == 8 * 16
    assert painted[0].min() >= 4 and painted[0].max() < 12


def test_alpha_split_unchanged():
    scene = _scene(
        [("minecraft:stone", {}, 0, 0, 0), ("minecraft:glass", {}, 1, 0, 0)]
    )
    img = _paint(scene, 1, _BG)
    glass = np.array([round(c * 0.8) for c in _GLASS[:3]] + [128], np.uint16)
    stone = np.array([round(c * 0.6) for c in _STONE[:3]] + [255], np.uint16)
    want = [round((glass[i] * 128 + stone[i] * 127) / 255) for i in range(3)]
    assert tuple(img[2, 2]) == (*want, 255)


def test_glass_behind_stone_fully_covered():
    scene = _scene(
        [("minecraft:glass", {}, 0, 0, 0), ("minecraft:stone", {}, 1, 0, 0)]
    )
    img = _paint(scene, 1, _BG)
    assert tuple(img[2, 2]) == (160, 24, 24, 255)


def test_cutout_texels_not_written():
    texture = np.full((16, 16, 4), (90, 90, 200, 255), np.uint8)
    texture[:, ::2, 3] = 0
    tex = _textures({"minecraft:stone": texture})
    img = _paint(_scene([("minecraft:stone", {}, 0, 0, 0)], tex), 4, _BG)
    painted = img[img[..., 3] > 0]
    allowed = {
        (90, 90, 200, 255),
        (72, 72, 160, 255),
        (54, 54, 120, 255),
    }
    assert {tuple(px) for px in painted.tolist()} <= allowed
    assert 0 < len(painted) < 192


def test_texel_rows_follow_face_orientation():
    texture = np.zeros((16, 16, 4), np.uint8)
    texture[:8] = (200, 30, 30, 255)
    texture[8:] = (30, 30, 200, 255)
    tex = _textures({"minecraft:stone": texture})
    img = _paint(_scene([("minecraft:stone", {}, 0, 0, 0)], tex), 4, _BG)
    assert tuple(img[1, 8]) == (200, 30, 30, 255)  # top far half, low v
    assert tuple(img[6, 8]) == (30, 30, 200, 255)  # top near half, high v
    assert tuple(img[6, 12]) == (120, 18, 18, 255)  # east upper: low v texel
    assert tuple(img[10, 12]) == (18, 18, 120, 255)  # east lower: high v texel


def test_magenta_sentinel_shaded_per_face():
    img = _paint(_scene([("minecraft:gold_block", {}, 0, 0, 0)]), 4, _BG)
    assert tuple(img[4, 8]) == (255, 0, 255, 255)
    assert tuple(img[10, 4]) == (204, 0, 204, 255)
    assert tuple(img[10, 12]) == (153, 0, 153, 255)


def test_same_depth_pair_paints_both():
    scene = _scene(
        [("minecraft:stone", {}, 1, 0, 0), ("minecraft:stone", {}, 0, 1, 0)]
    )
    img = _paint(scene, 1, _BG)
    assert int((img[..., 3] != 0).sum()) == 24


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


def test_deterministic_bytes():
    blocks = [
        ("minecraft:stone", {}, 0, 0, 0),
        ("minecraft:glass", {}, 1, 0, 0),
        ("minecraft:stone", {}, 0, 1, 1),
    ]
    first = _paint(_scene(blocks), 3, _BG)
    second = _paint(_scene(blocks), 3, _BG)
    assert first.tobytes() == second.tobytes()


def test_background_fills_canvas():
    img = _paint(
        _scene([("minecraft:stone", {}, 0, 0, 0)]), 1, (255, 255, 255, 255)
    )
    assert (img[..., 3] == 255).all()


def test_empty_scene_paints_background():
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
    img = _paint(_compose(schem, 0, TEX), 2, _BG)
    assert img.shape == (8, 8, 4)
    assert (img == 0).all()


def test_negative_projection_stays_off_canvas():
    quads = np.empty(1, QUAD_DTYPE)
    quads["vtx"] = [[(-8, 16, 16), (8, 16, 16), (8, 0, 16), (-8, 0, 16)]]
    quads["uv"] = [[(0, 0), (16, 0), (16, 16), (0, 16)]]
    quads["texkey"] = "block/odd"
    quads["shade"] = TIER_SOUTH
    quads["dbl"] = False
    pack = _raw_quad_pack(quads, _ODD)
    scene = _scene([("minecraft:odd", {}, 0, 0, 0)], pack)
    img = _paint(scene, 2, _BG)
    assert int((img[..., 3] != 0).sum()) == 8
    assert set(np.nonzero(img[..., 3])[1].tolist()) == {0, 1}
    painted = {tuple(px) for px in img.reshape(-1, 4).tolist() if px[3]}
    assert painted == {(72, 160, 72, 255)}


def test_overflow_projection_clipped_silently():
    quads = np.empty(1, QUAD_DTYPE)
    quads["vtx"] = [[(32, 16, 16), (48, 16, 16), (48, 0, 16), (32, 0, 16)]]
    quads["uv"] = [[(0, 0), (16, 0), (16, 16), (0, 16)]]
    quads["texkey"] = "block/odd"
    quads["shade"] = TIER_SOUTH
    quads["dbl"] = False
    pack = _raw_quad_pack(quads, _ODD)
    scene = _scene([("minecraft:odd", {}, 0, 0, 0)], pack)
    img = _paint(scene, 2, _BG)
    assert (img == 0).all()


def test_non_parallelogram_quad_fails_fast():
    quads = np.empty(1, QUAD_DTYPE)
    quads["vtx"] = [[(0, 16, 16), (16, 16, 16), (12, 0, 16), (0, 0, 16)]]
    quads["uv"] = [[(0, 0), (16, 0), (16, 16), (0, 16)]]
    quads["texkey"] = "block/odd"
    quads["shade"] = TIER_SOUTH
    quads["dbl"] = False
    pack = _raw_quad_pack(quads, _ODD)
    scene = _scene([("minecraft:odd", {}, 0, 0, 0)], pack)
    with pytest.raises(RenderError, match="parallelogram"):
        _paint(scene, 2, _BG)


def test_semi_transparent_alpha_preserved():
    img = _paint(_scene([("minecraft:glass", {}, 0, 0, 0)]), 1, _BG)
    assert (img[..., 3] == 128).sum() == 12


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
        (255, 0, 255, 255),
        (204, 0, 204, 255),
        (153, 0, 153, 255),
    }
    flat = {tuple(px) for px in img.reshape(-1, 4).tolist() if px[3]}
    assert not (flat & magenta)
