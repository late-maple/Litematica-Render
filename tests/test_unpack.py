"""Bit-unpacking contract: roundtrip, cross-long boundaries, bit width, YZX order."""

import numpy as np
import pytest
from helpers.litematica_packer import pack_indices

from litematic.model import Vec3i
from litematic.parser import _bits_for, _unpack_indices


def test_bits_for_boundaries():
    assert _bits_for(1) == 2
    assert _bits_for(2) == 2
    assert _bits_for(3) == 2
    assert _bits_for(4) == 2
    assert _bits_for(5) == 3
    assert _bits_for(8) == 3
    assert _bits_for(9) == 4
    assert _bits_for(16) == 4
    assert _bits_for(17) == 5


def test_yzx_layout_known_values():
    # 3x2x4 (lenY != lenZ, so a swapped reshape shows up), palette of 7 -> 3
    # bits; 24 entries span two longs and entry 21 straddles the boundary
    size = Vec3i(3, 2, 4)
    values_yzx = np.array(
        [0, 1, 2, 3, 4, 5, 6, 0, 2, 4, 6, 1, 3, 5, 0, 3, 6, 2, 5, 1, 4, 0, 4, 1],
        dtype=np.uint16,
    )
    packed = pack_indices(values_yzx, palette_size=7)
    out = _unpack_indices(packed, bits=3, size=size)
    assert out.shape == (3, 2, 4)
    assert out.dtype == np.uint16
    for x in range(size.x):
        for y in range(size.y):
            for z in range(size.z):
                file_index = (y * size.z + z) * size.x + x
                assert out[x, y, z] == values_yzx[file_index]


def test_entry_crossing_long_boundary():
    # bits=5: 13 entries fit in 65 bits -> entry 12 straddles longs 0 and 1
    size = Vec3i(13, 1, 1)
    values_yzx = np.arange(13, dtype=np.uint16) % 32
    values_yzx[12] = 0b10111
    packed = pack_indices(values_yzx, palette_size=32)
    out = _unpack_indices(packed, bits=5, size=size)
    assert int(out[12, 0, 0]) == 0b10111
    np.testing.assert_array_equal(out.reshape(-1), values_yzx)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 8, 11, 15])
@pytest.mark.parametrize(
    "size",
    [Vec3i(1, 1, 1), Vec3i(7, 3, 5), Vec3i(16, 16, 16), Vec3i(3, 1, 13)],
)
def test_roundtrip_random(bits, size):
    rng = np.random.default_rng(42)
    palette_size = 1 << bits
    count = size.x * size.y * size.z
    values_yzx = rng.integers(0, palette_size, size=count).astype(np.uint16)
    packed = pack_indices(values_yzx, palette_size)
    out = _unpack_indices(packed, bits=bits, size=size)
    assert out.dtype == np.uint16
    # expected built straight from the contract formula, independent of the
    # implementation's reshape/transpose
    expected = np.empty((size.x, size.y, size.z), dtype=np.uint16)
    for y in range(size.y):
        for z in range(size.z):
            for x in range(size.x):
                expected[x, y, z] = values_yzx[(y * size.z + z) * size.x + x]
    np.testing.assert_array_equal(out, expected)


def test_wide_palette_uses_uint32():
    # 17 bits needed for a 100000-entry palette -> uint32 output;
    # entry 0 fills the low bits of the first long
    size = Vec3i(4, 1, 1)
    packed = np.array([100000 - 1, 0], dtype=np.uint64).view(np.int64)
    out = _unpack_indices(packed, bits=17, size=size)
    assert out.dtype == np.uint32
    assert int(out[0, 0, 0]) == 100000 - 1
