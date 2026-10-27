"""Litematica palette-index bit unpacking."""

import numpy as np

from litematic.model import Vec3i


def _bits_for(palette_size: int) -> int:
    """Bit width per entry: enough for palette_size values, at least 2."""
    return max(2, (palette_size - 1).bit_length())


def _unpack_indices(packed: np.ndarray, bits: int, size: Vec3i) -> np.ndarray:
    """Unpack litematica's continuous bit stream into an (x, y, z) index array.

    Entry i starts at bit i*bits of the long array, filling each long upward
    from its low bit; an entry straddling a boundary continues into the low
    bits of the next long. File order is
    YZX: index = (y*lenZ + z)*lenX + x. Palettes wider than 16 bits need
    uint32 output.
    """
    count = size.x * size.y * size.z
    words = packed.view(np.uint64)  # logical shifts; int64 storage is untouched
    positions = np.arange(count, dtype=np.uint64) * np.uint64(bits)
    long_index = positions >> np.uint64(6)
    bit_offset = positions & np.uint64(63)
    lo = words[long_index] >> bit_offset
    # bits taken from the next long; only straddling entries shift at all,
    # and for them the shift is always in [1, 63]
    spill = bit_offset.astype(np.int64) + bits - 64
    crossing = spill > 0
    hi = np.zeros(count, dtype=np.uint64)
    if crossing.any():
        idx = np.nonzero(crossing)[0]
        hi[idx] = words[long_index[idx] + 1] << (np.uint64(64) - bit_offset[idx])
    mask = (1 << bits) - 1
    values = (lo | hi) & np.uint64(mask)
    dtype = np.uint16 if bits <= 16 else np.uint32
    return values.astype(dtype).reshape(size.y, size.z, size.x).transpose(2, 0, 1)
