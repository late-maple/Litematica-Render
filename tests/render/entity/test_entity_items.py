"""Dropped-item entity builders: mini-block scaling and flat sprite quads."""

import numpy as np
import pytest

from litematic.render.entity import (
    ITEM_BLOCK_SCALE,
    ITEM_FLAT_SIZE,
    ITEM_MINI_BASE,
    item_entity_quads,
    magenta_entity_quads,
)
from litematic.render.pack.resolve import Resolver
from litematic.render.quads import CULL_NONE, unit_cube_quads


def _textures(item_display, quads_map=None, arrays=None):
    return Resolver(
        version="pack-quads.1",
        arrays=arrays or {},
        quads_map=quads_map or {},
        occl_map={},
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map={},
        item_display=item_display,
    )


def _cube_textures(entry):
    return _textures(
        entry,
        quads_map={"minecraft:stone[]": unit_cube_quads("block/stone")},
        arrays={"block/stone": np.zeros((16, 16, 4), np.uint8)},
    )


def test_flat_item_quad_geometry_pinned():
    """Generated ground display: half-size vertical sprite, bob-mid hover."""
    textures = _textures(
        {"minecraft:wheat_seeds": {"flat": "item/wheat_seeds"}},
        arrays={"item/wheat_seeds": np.zeros((16, 16, 4), np.uint8)},
    )
    quads = item_entity_quads("minecraft:wheat_seeds", textures)
    assert quads is not None and len(quads) == 1
    assert str(quads["texkey"][0]) == "item/wheat_seeds"
    assert bool(quads["dbl"][0])
    assert int(quads["cull"][0]) == CULL_NONE
    x = quads["vtx"][0][:, 0]
    y = quads["vtx"][0][:, 1]
    z = quads["vtx"][0][:, 2]
    assert x.min() == 8 - ITEM_FLAT_SIZE / 2 and x.max() == 8 + ITEM_FLAT_SIZE / 2
    assert y.min() == pytest.approx(ITEM_MINI_BASE)
    assert y.max() == pytest.approx(ITEM_MINI_BASE + ITEM_FLAT_SIZE)
    assert (z == 8).all()


def test_block_item_quads_scale_to_mini_cube():
    textures = _cube_textures({"minecraft:stone": {"block": "minecraft:stone"}})
    quads = item_entity_quads("minecraft:stone", textures)
    assert quads is not None and len(quads) == 6
    v = quads["vtx"]
    assert v[..., 0].min() == pytest.approx(8 - 8 * ITEM_BLOCK_SCALE)
    assert v[..., 0].max() == pytest.approx(8 + 8 * ITEM_BLOCK_SCALE)
    assert v[..., 1].min() == pytest.approx(ITEM_MINI_BASE)
    assert v[..., 1].max() == pytest.approx(ITEM_MINI_BASE + 16 * ITEM_BLOCK_SCALE)
    assert (quads["dbl"]).all()
    assert (quads["cull"] == CULL_NONE).all()
    assert {str(k) for k in quads["texkey"]} == {"block/stone"}


def test_unknown_item_and_missing_entries_return_none():
    textures = _textures({})
    assert item_entity_quads("minecraft:ghost", textures) is None
    assert item_entity_quads(
        "minecraft:stone", _textures({"minecraft:stone": {"block": "minecraft:stone"}})
    ) is None
    assert item_entity_quads(
        "minecraft:x",
        _textures({"minecraft:x": {"flat": "item/missing"}}),
    ) is None


def test_magenta_entity_quads_pin_fallback():
    quads = magenta_entity_quads()
    assert len(quads) == 1
    assert str(quads["texkey"][0]) == "__magenta__"
    assert bool(quads["dbl"][0])
    v = quads["vtx"][0]
    assert v[:, 0].min() == pytest.approx(4.0)
    assert v[:, 0].max() == pytest.approx(12.0)
    assert v[:, 1].min() == pytest.approx(4.0)
    assert v[:, 1].max() == pytest.approx(12.0)
    assert (v[:, 2] == 15.0).all()


def test_texture_only_pack_falls_back_to_direct_item_texture():
    """No models at all: the direct sprite renders (old Task-1 gap fix)."""
    textures = _textures(
        {},
        arrays={"item/wheat_seeds": np.zeros((16, 16, 4), np.uint8)},
    )
    quads = item_entity_quads("minecraft:wheat_seeds", textures)
    assert quads is not None
    assert {str(k) for k in quads["texkey"]} == {"item/wheat_seeds"}
