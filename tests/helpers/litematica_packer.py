"""Reference bit-stream packer, independent from the product unpacker.

Packs YZX-ordered palette indices into litematica's continuous bit stream:
entry i is placed at bit i*bits of the long array, filling each long upward
from its low bit and spilling into the next long when it straddles a boundary.
"""

import numpy as np


def pack_indices(indices_yzx: np.ndarray, palette_size: int) -> np.ndarray:
    bits = max(2, (palette_size - 1).bit_length())
    flat = indices_yzx.reshape(-1)
    n_longs = (len(flat) * bits + 63) // 64
    longs = [0] * n_longs
    mask = (1 << bits) - 1
    u64 = (1 << 64) - 1
    for i, value in enumerate(flat):
        g = i * bits
        li, off = g >> 6, g & 63
        # explicit 64-bit wraparound; Python ints will not wrap on their own
        longs[li] = (longs[li] | ((int(value) & mask) << off)) & u64
        spill = off + bits - 64
        if spill > 0:
            longs[li + 1] |= (int(value) & mask) >> (bits - spill)
    return np.array(longs, dtype=np.uint64).view(np.int64)
