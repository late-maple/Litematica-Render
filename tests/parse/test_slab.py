"""Region.slab_indices contract: lazy per-slab unpack, no caching."""

import numpy as np
import pytest
from helpers.litematica_packer import build_litematic_bytes

from litematic import LitematicFormatError, loads
from litematic.parse.model import BlockState, Region, Vec3i


def _mixed_region():
    palette = [
        ("minecraft:air", {}),
        ("minecraft:stone", {}),
        ("minecraft:dirt", {}),
        ("minecraft:glass", {}),
    ]
    # Asymmetric 3x2x4 with varied entries so every slab row differs.
    indices = np.arange(3 * 2 * 4, dtype=np.uint16) % 4
    data = build_litematic_bytes(
        [
            {
                "name": "Main",
                "position": (0, 0, 0),
                "size": (3, 2, 4),
                "palette": palette,
                "indices_yzx": indices,
            }
        ]
    )
    return loads(data).regions["Main"]


@pytest.mark.parametrize("y0, y1", [(0, 1), (1, 2), (0, 2)])
def test_slab_matches_full_indices(y0, y1):
    region = _mixed_region()
    slab = region.slab_indices(y0, y1)
    assert np.array_equal(slab, region.block_indices[:, y0:y1, :])


def test_slab_dtype_follows_palette_width():
    size = (1, 1, 1)
    palette = [("minecraft:air", {})] + [
        (f"minecraft:b{i}", {}) for i in range(65536)
    ]
    data = build_litematic_bytes(
        [
            {
                "name": "Main",
                "position": (0, 0, 0),
                "size": size,
                "palette": palette,
                "indices_yzx": np.array([1], dtype=np.uint32),
            }
        ]
    )
    region = loads(data).regions["Main"]
    assert region.slab_indices(0, 1).dtype == np.uint32


def test_slab_bounds_raise_index_error():
    region = _mixed_region()
    with pytest.raises(IndexError):
        region.slab_indices(1, 1)
    with pytest.raises(IndexError):
        region.slab_indices(0, 3)
    with pytest.raises(IndexError):
        region.slab_indices(-1, 1)


def test_slab_palette_overflow_names_region():
    data = build_litematic_bytes(
        [
            {
                "name": "Main",
                "position": (0, 0, 0),
                "size": (1, 2, 1),
                "palette": [("minecraft:air", {}), ("minecraft:stone", {})],
                # Two 2-bit entries: row 0 -> 1 (valid), row 1 -> 2 (overflow).
                "packed": np.array([1 | (2 << 2)], dtype=np.int64),
            }
        ]
    )
    region = loads(data).regions["Main"]
    assert int(region.slab_indices(0, 1)[0, 0, 0]) == 1
    with pytest.raises(LitematicFormatError, match="Main"):
        region.slab_indices(1, 2)


def test_slab_not_cached_and_readonly():
    region = _mixed_region()
    first = region.slab_indices(0, 1)
    second = region.slab_indices(0, 1)
    assert first is not second
    assert not first.flags.writeable
    assert not region.is_unpacked


def test_slab_falls_back_without_injected_unpacker():
    region = Region(
        name="Main",
        position=Vec3i(0, 0, 0),
        size=Vec3i(2, 3, 4),
        palette=[BlockState("minecraft:air", {})],
        block_entities=(),
        entities=(),
        pending_block_ticks=None,
        pending_fluid_ticks=None,
        packed_indices=np.zeros(8, dtype=np.int64),
        _unpack=lambda: np.zeros((2, 3, 4), dtype=np.uint16),
    )
    slab = region.slab_indices(1, 3)
    assert slab.shape == (2, 2, 4)
    assert not slab.flags.writeable
