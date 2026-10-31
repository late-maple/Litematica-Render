"""Armor stand builders: vanilla part boxes, small scale, toggles."""

import numpy as np
import pytest

from litematic.render.entity import armor_stand_quads, entity_props
from litematic.render.pack.resolve import Resolver


def _textures(**display):
    return Resolver(
        version="pack-quads.1",
        arrays={"entity/armorstand/armorstand": np.zeros((64, 64, 4), np.uint8)},
        quads_map={},
        occl_map={},
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map={},
        item_display=display,
    )


def _bounds(quads):
    v = quads["vtx"]
    return (
        float(v[..., 0].min()),
        float(v[..., 0].max()),
        float(v[..., 1].min()),
        float(v[..., 1].max()),
    )


def test_stand_default_build_extents_pinned():
    quads = armor_stand_quads({}, _textures())
    x0, x1, y0, y1 = _bounds(quads)
    assert y1 == pytest.approx(30.0)  # vanilla model-table 30-texel height
    assert y0 == pytest.approx(0.0)  # base plate on the ground
    assert x0 == pytest.approx(2.0) and x1 == pytest.approx(14.0)


def test_stand_show_arms_adds_arm_boxes():
    plain = armor_stand_quads({}, _textures())
    armed = armor_stand_quads({"ShowArms": True}, _textures())
    assert _bounds(plain)[0] == pytest.approx(2.0)
    assert _bounds(armed)[0] == pytest.approx(1.0)  # arms reach x=1
    assert len(armed) > len(plain)


def test_stand_no_base_plate_drops_the_plate():
    quads = armor_stand_quads({"NoBasePlate": True}, _textures())
    assert _bounds(quads)[2] == pytest.approx(1.0)  # legs start at y=1


def test_stand_small_scales_to_the_baby_transform():
    quads = armor_stand_quads({"Small": True}, _textures())
    _, _, _, y1 = _bounds(quads)
    assert y1 == pytest.approx(16.5)
    assert _bounds(quads)[2] == pytest.approx(0.0)


def test_stand_invisible_and_missing_texture_return_none():
    assert armor_stand_quads({"Invisible": True}, _textures()) is None
    empty = Resolver(
        version="v", arrays={}, quads_map={}, occl_map={},
        pack_mtime=0.0, states=None, parts_map={}, mquad_map={},
    )
    assert armor_stand_quads({}, empty) is None


def test_stand_yaw_rotates_about_the_cell_center():
    quads = armor_stand_quads({"ShowArms": True, "yaw": 90.0}, _textures())
    _, _, _, y1 = _bounds(quads)
    v = quads["vtx"]
    assert y1 == pytest.approx(30.0)
    assert float(v[..., 2].min()) == pytest.approx(1.0)  # arms swung to +z


def test_entity_props_normalizes_real_nbt():
    props = entity_props(
        {"Facing": 3, "Rotation": [45.0, 0.0], "Invisible": 1,
         "ItemRotation": 3}
    )
    assert props["facing"] == "south"
    assert props["yaw"] == 45.0
    assert props["Invisible"] is True
    assert props["item_rotation"] == 3
    assert entity_props({"Facing": 2})["facing"] == "north"
    assert entity_props({})["facing"] is None
