"""Item frame builders: pad orientation, content rotation, fallbacks."""

import numpy as np
import pytest

from litematic.render.entity import item_frame_quads
from litematic.render.pack.resolve import Resolver
from litematic.render.quads import QUAD_DTYPE


def _pad():
    quads = np.zeros(2, QUAD_DTYPE)
    quads["texkey"][:] = "block/item_frame"
    quads["vtx"][:, :, 2] = 15.5
    quads["vtx"][:, 0, 0] = 2.0
    quads["vtx"][:, 1, 0] = 14.0
    return quads


def _textures(item_display=None, arrays=None, mquad_map=None):
    return Resolver(
        version="pack-quads.1",
        arrays=arrays or {},
        quads_map={},
        occl_map={},
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map=mquad_map or {},
        item_display=item_display,
    )


def _frame_textures(extra_display=None):
    display = {"minecraft:diamond": {"flat": "item/diamond"}}
    display.update(extra_display or {})
    return _textures(
        display,
        arrays={
            "item/diamond": np.zeros((16, 16, 4), np.uint8),
            "block/item_frame": np.zeros((16, 16, 4), np.uint8),
            "block/stone": np.zeros((16, 16, 4), np.uint8),
        },
        mquad_map={"__frame_pad": _pad()},
    )


def _texkeys(quads):
    return {str(k) for k in quads["texkey"]}


def test_frame_north_keeps_base_pad_plane_and_content_flush():
    quads = item_frame_quads(
        {"facing": "north", "item_rotation": 0},
        "minecraft:diamond",
        _frame_textures(),
    )
    pad = quads[np.isin(quads["texkey"], ["block/item_frame"])]
    content = quads[quads["texkey"] == "item/diamond"]
    assert len(pad) == 2 and len(content) == 1
    assert (content["vtx"][0][:, 2] == 15.0).all()
    assert content["vtx"][0][:, 0].min() == pytest.approx(4.0)
    assert content["vtx"][0][:, 0].max() == pytest.approx(12.0)
    assert content["vtx"][0][:, 1].min() == pytest.approx(4.0)
    assert content["vtx"][0][:, 1].max() == pytest.approx(12.0)


def test_frame_content_rotates_in_45_degree_steps():
    quads = item_frame_quads(
        {"facing": "north", "item_rotation": 1},
        "minecraft:diamond",
        _frame_textures(),
    )
    content = quads[quads["texkey"] == "item/diamond"]
    angle = np.deg2rad(45.0)
    dx, dy = 4.0 - 8.0, 12.0 - 8.0
    cos, sin = np.cos(angle), np.sin(angle)
    corner = content["vtx"][0][0]
    assert corner[0] == pytest.approx(8 + dx * cos - dy * sin)
    assert corner[1] == pytest.approx(8 + dx * sin + dy * cos)


def test_frame_invisible_renders_content_only():
    quads = item_frame_quads(
        {"facing": "south", "item_rotation": 0, "Invisible": True},
        "minecraft:diamond",
        _frame_textures(),
    )
    assert _texkeys(quads) == {"item/diamond"}


def test_frame_without_item_renders_pad_only():
    quads = item_frame_quads({"facing": "south"}, None, _frame_textures())
    assert _texkeys(quads) == {"block/item_frame"}


def test_frame_block_content_uses_block_texture_plate():
    textures = _frame_textures(
        {"minecraft:stone": {"block": "minecraft:stone"}}
    )
    textures.quads_map["minecraft:stone[]"] = np.zeros(1, QUAD_DTYPE)
    textures.quads_map["minecraft:stone[]"]["texkey"] = "block/stone"
    quads = item_frame_quads(
        {"facing": "south", "item_rotation": 0},
        "minecraft:stone",
        textures,
    )
    assert "block/stone" in _texkeys(quads)


def test_frame_unresolvable_content_renders_magenta():
    quads = item_frame_quads(
        {"facing": "south", "item_rotation": 0},
        "minecraft:ghost",
        _frame_textures(),
    )
    assert "__magenta__" in _texkeys(quads)


def test_frame_vertical_facing_and_missing_pad_return_none():
    textures = _frame_textures()
    assert item_frame_quads({"facing": "up"}, None, textures) is None
    assert item_frame_quads({"facing": "down"}, None, textures) is None
    assert item_frame_quads({"facing": "south"}, None, _textures()) is None


def test_frame_content_follows_facing_turn():
    """Pad hugs the wall opposite the facing; content flush with its front."""
    for facing, plane_axis, plane in (
        ("south", 2, 1.0),
        ("north", 2, 15.0),
        ("east", 0, 1.0),
        ("west", 0, 15.0),
    ):
        quads = item_frame_quads(
            {"facing": facing, "item_rotation": 0},
            "minecraft:diamond",
            _frame_textures(),
        )
        content = quads[quads["texkey"] == "item/diamond"]
        assert (content["vtx"][0][:, plane_axis] == plane).all(), facing


def test_frame_pad_falls_back_to_frame_sprite_for_texture_only_packs():
    textures = _textures(
        {"minecraft:diamond": {"flat": "item/diamond"}},
        arrays={
            "item/diamond": np.zeros((16, 16, 4), np.uint8),
            "item/item_frame": np.zeros((16, 16, 4), np.uint8),
        },
    )
    quads = item_frame_quads(
        {"facing": "north", "item_rotation": 0},
        "minecraft:diamond",
        textures,
    )
    keys = _texkeys(quads)
    assert "item/item_frame" in keys and "item/diamond" in keys
    pad = quads[quads["texkey"] == "item/item_frame"]
    assert (pad["vtx"][0][:, 2] == 15.5).all()  # wall plane stand-in


def test_tier_turns_match_vertex_turn_geometry():
    from litematic.render.entity import _TIER_TURNS

    normals = {
        0: (0, -1, 0),
        1: (0, 1, 0),
        2: (0, 0, -1),
        3: (0, 0, 1),
        4: (-1, 0, 0),
        5: (1, 0, 0),
    }
    to_tier = {v: k for k, v in normals.items()}
    for turns in (1, 2, 3):
        assert (_TIER_TURNS[turns][0], _TIER_TURNS[turns][1]) == (0, 1)
        for tier in (2, 3, 4, 5):
            normal = normals[tier]
            dx, dz = normal[0], normal[2]
            if turns == 1:
                nx, nz = dz, -dx
            elif turns == 2:
                nx, nz = -dx, -dz
            else:
                nx, nz = -dz, dx
            assert _TIER_TURNS[turns][tier] == to_tier[(nx, 0, nz)]
