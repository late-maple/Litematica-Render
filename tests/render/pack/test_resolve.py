"""Block resolver contract: the per-state resolution chain at one seam."""

import json

import numpy as np
import pytest

from litematic.parse.model import BlockState
from litematic.render.errors import RenderError
from litematic.render.pack.resolve import (
    MISS_NO_GEOMETRY,
    MISS_NO_TEXTURE,
    Resolver,
)
from litematic.render.quads import unit_cube_quads

_STONE = (140, 140, 140, 255)
_GRASS_TINT = (145, 189, 89)


def _resolver(**overrides):
    fields = dict(
        version="pack-quads",
        arrays={"block/stone": np.full((16, 16, 4), _STONE, np.uint8)},
        quads_map={"minecraft:stone[]": unit_cube_quads("block/stone")},
        occl_map={"minecraft:stone[]": True},
        pack_mtime=0.0,
        states=None,
        parts_map={},
        mquad_map={},
    )
    fields.update(overrides)
    return Resolver(**fields)


def _state(name, properties=None):
    return BlockState(name, properties or {})


def test_pack_hit_resolves_quads_occludes_miss_none():
    resolver = _resolver()
    res = resolver.resolve(_state("minecraft:stone"))
    assert res.miss is None
    assert res.occludes
    assert len(res.quads) == 6
    assert set(str(k) for k in res.quads["texkey"]) == {"block/stone"}


def test_bare_key_fallback_ignores_extra_properties():
    resolver = _resolver()
    res = resolver.resolve(_state("minecraft:stone", {"waterlogged": "false"}))
    assert res.miss is None
    assert len(res.quads) == 6


def test_indexed_states_require_pair_match():
    quads = unit_cube_quads("block/thing")
    resolver = _resolver(
        arrays={"block/thing": np.zeros((16, 16, 4), np.uint8)},
        quads_map={"minecraft:thing[facing=north]": quads},
        occl_map={"minecraft:thing[facing=north]": True},
        states={"minecraft:thing": {"facing=north": {"facing": "north"}}},
    )
    hit = resolver.resolve(
        _state("minecraft:thing", {"facing": "north", "waterlogged": "false"})
    )
    assert hit.miss is None
    miss = resolver.resolve(_state("minecraft:thing", {"facing": "south"}))
    assert miss.miss == MISS_NO_GEOMETRY
    assert miss.occludes is False


def test_multipart_evaluated_per_state():
    post = unit_cube_quads("block/post")
    arm = unit_cube_quads("block/arm")
    resolver = _resolver(
        arrays={
            "block/post": np.zeros((16, 16, 4), np.uint8),
            "block/arm": np.zeros((16, 16, 4), np.uint8),
        },
        parts_map={
            "minecraft:flagpost": json.dumps(
                {
                    "parts": [
                        {"when": None, "quads": "post"},
                        {"when": {"north": "true"}, "quads": "arm"},
                    ]
                }
            )
        },
        mquad_map={"post": post, "arm": arm},
    )
    off = resolver.resolve(_state("minecraft:flagpost", {"north": "false"}))
    assert off.miss is None
    assert set(str(k) for k in off.quads["texkey"]) == {"block/post"}
    on = resolver.resolve(_state("minecraft:flagpost", {"north": "true"}))
    assert set(str(k) for k in on.quads["texkey"]) == {"block/post", "block/arm"}


def test_multipart_all_conditions_false_falls_through():
    arm = unit_cube_quads("block/arm")
    resolver = _resolver(
        parts_map={
            "minecraft:stub": json.dumps(
                {"parts": [{"when": {"north": "true"}, "quads": "arm"}]}
            )
        },
        mquad_map={"arm": arm},
    )
    res = resolver.resolve(_state("minecraft:stub", {"north": "false"}))
    assert res.miss == MISS_NO_GEOMETRY
    assert set(str(k) for k in res.quads["texkey"]) == {"__magenta__"}


def test_wire_multipart_rewrites_to_live_power_tint():
    dot = unit_cube_quads("block/redstone_dust_dot")
    resolver = _resolver(
        arrays={
            "block/redstone_dust_dot@255,26,26": np.zeros(
                (16, 16, 4), np.uint8
            ),
            "block/redstone_dust_dot@76,0,0": np.zeros((16, 16, 4), np.uint8),
        },
        parts_map={
            "minecraft:redstone_wire": json.dumps(
                {"parts": [{"when": None, "quads": "dot"}]}
            )
        },
        mquad_map={"dot": dot},
    )
    full = resolver.resolve(_state("minecraft:redstone_wire", {"power": "15"}))
    assert set(str(k) for k in full.quads["texkey"]) == {
        "block/redstone_dust_dot@255,26,26"
    }
    off = resolver.resolve(_state("minecraft:redstone_wire", {"power": "0"}))
    assert set(str(k) for k in off.quads["texkey"]) == {
        "block/redstone_dust_dot@76,0,0"
    }


def test_wire_multipart_keeps_key_when_tinted_array_missing():
    dot = unit_cube_quads("block/redstone_dust_dot@9,9,9")
    resolver = _resolver(
        parts_map={
            "minecraft:redstone_wire": json.dumps(
                {"parts": [{"when": None, "quads": "dot"}]}
            )
        },
        mquad_map={"dot": dot},
    )
    res = resolver.resolve(_state("minecraft:redstone_wire", {"power": "15"}))
    assert set(str(k) for k in res.quads["texkey"]) == {
        "block/redstone_dust_dot@9,9,9"
    }


def test_builtin_backfill_direct_key():
    resolver = _resolver(
        arrays={"block/oak_fence": np.zeros((16, 16, 4), np.uint8)},
    )
    res = resolver.resolve(_state("minecraft:oak_fence"))
    assert res.miss is None
    assert res.occludes is False
    assert set(str(k) for k in res.quads["texkey"]) == {"block/oak_fence"}


def test_builtin_backfill_tinted_key_wins():
    resolver = _resolver(
        arrays={
            f"block/short_grass@{_GRASS_TINT[0]},{_GRASS_TINT[1]},{_GRASS_TINT[2]}": (
                np.full((16, 16, 4), (1, 2, 3, 255), np.uint8)
            ),
            "block/short_grass": np.full((16, 16, 4), (9, 9, 9, 255), np.uint8),
        },
    )
    res = resolver.resolve(_state("minecraft:short_grass"))
    assert res.miss is None
    assert set(str(k) for k in res.quads["texkey"]) == {
        "block/short_grass@145,189,89"
    }


def test_builtin_backfill_wood_material_key():
    resolver = _resolver(
        arrays={"block/oak_planks": np.zeros((16, 16, 4), np.uint8)},
    )
    res = resolver.resolve(_state("minecraft:oak_fence"))
    assert res.miss is None
    assert set(str(k) for k in res.quads["texkey"]) == {"block/oak_planks"}


def test_builtin_backfill_door_top_bottom():
    resolver = _resolver(
        arrays={
            "block/oak_door_top": np.zeros((16, 16, 4), np.uint8),
            "block/oak_door_bottom": np.zeros((16, 16, 4), np.uint8),
        },
    )
    res = resolver.resolve(_state("minecraft:oak_door"))
    assert res.miss is None
    tops = res.quads["vtx"][..., 1].mean(axis=1) >= 8.0
    assert set(str(k) for k in res.quads["texkey"][tops]) == {
        "block/oak_door_top"
    }
    assert set(str(k) for k in res.quads["texkey"][~tops]) == {
        "block/oak_door_bottom"
    }


def test_builtin_family_without_texture_misses_no_texture():
    resolver = _resolver(arrays={})
    res = resolver.resolve(_state("minecraft:oak_fence"))
    assert res.miss == MISS_NO_TEXTURE
    assert set(str(k) for k in res.quads["texkey"]) == {"__magenta__"}
    assert res.occludes is False


def test_unknown_block_misses_no_geometry_magenta_cube():
    resolver = _resolver()
    res = resolver.resolve(_state("minecraft:create:x"))
    assert res.miss == MISS_NO_GEOMETRY
    assert len(res.quads) == 6
    assert set(str(k) for k in res.quads["texkey"]) == {"__magenta__"}
    assert res.occludes is False


def test_texkey_arrays_returns_requested_arrays():
    resolver = _resolver()
    out = resolver.texkey_arrays(["block/stone", "__magenta__"])
    assert set(out) == {"block/stone", "__magenta__"}
    assert (out["block/stone"] == np.full((16, 16, 4), _STONE, np.uint8)).all()
    assert out["__magenta__"].shape == (16, 16, 4)


def test_texkey_arrays_unknown_key_raises():
    resolver = _resolver()
    with pytest.raises(RenderError, match="texkey missing"):
        resolver.texkey_arrays(["block/ghost"])
