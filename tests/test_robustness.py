"""Robustness: corrupt and degenerate inputs fail fast as LitematicError."""

import numpy as np
import pytest
from helpers.litematica_packer import build_litematic_bytes

from litematic import LitematicFormatError, NbtError, loads


def _valid_bytes():
    return build_litematic_bytes(
        [
            {
                "name": "Main",
                "position": (0, 0, 0),
                "size": (2, 2, 2),
                "palette": [("minecraft:air", {}), ("minecraft:stone", {})],
                "indices_yzx": np.zeros(8, dtype=np.uint16),
            }
        ]
    )


def test_truncated_file_raises_nbt_error():
    data = _valid_bytes()
    with pytest.raises(NbtError):
        loads(data[: len(data) // 2])


def test_leading_garbage_raises_nbt_error():
    with pytest.raises(NbtError):
        loads(b"XX" + _valid_bytes())


def test_huge_palette_short_array_fails_fast_at_load():
    # palette of 100000 -> 17 bits; 4x4x4 blocks need 17 longs, not 1
    palette = [("minecraft:air", {})] * 100000
    build = build_litematic_bytes(
        [
            {
                "name": "Main",
                "position": (0, 0, 0),
                "size": (4, 4, 4),
                "palette": palette,
                "packed": np.zeros(1, dtype=np.int64),
            }
        ]
    )
    with pytest.raises(LitematicFormatError):
        loads(build)


def test_all_air_single_block_region_parses():
    data = build_litematic_bytes(
        [
            {
                "name": "Main",
                "position": (0, 0, 0),
                "size": (1, 1, 1),
                "palette": [("minecraft:air", {})],
                "indices_yzx": np.zeros(1, dtype=np.uint16),
            }
        ]
    )
    region = loads(data).regions["Main"]
    indices = region.block_indices
    assert indices.shape == (1, 1, 1)
    assert int(indices[0, 0, 0]) == 0
    assert region.get_block(0, 0, 0).name == "minecraft:air"
