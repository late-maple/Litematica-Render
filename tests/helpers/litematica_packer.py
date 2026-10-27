"""Reference bit-stream packer, independent from the product unpacker.

Packs YZX-ordered palette indices into litematica's continuous bit stream:
entry i is placed at bit i*bits of the long array, filling each long upward
from its low bit and spilling into the next long when it straddles a boundary.
"""

import numpy as np

from helpers.nbt_writer import write_nbt


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


def build_litematic_bytes(
    regions: list[dict],
    *,
    metadata_overrides: dict | None = None,
    data_version: int = 3337,
    extra_root: dict | None = None,
    gzip: bool = True,
) -> bytes:
    """Assemble a complete .litematic byte blob for parser tests.

    Each region dict: name, position, size, palette (list of (name, props)),
    indices_yzx OR packed (long array used verbatim), plus optional
    tile_entities, entities, pending_block_ticks, pending_fluid_ticks.
    Metadata is written verbatim from metadata_overrides (no invented fields),
    so tests control exactly which keys the file carries.
    """
    meta = dict(metadata_overrides or {})
    regions_nbt: dict = {}
    for r in regions:
        palette = r.get("palette")
        palette_entries = []
        if palette is not None:
            for name, props in palette:
                entry = {"Name": name}
                if props:
                    entry["Properties"] = props
                palette_entries.append(entry)
        packed = r.get("packed")
        if packed is None and palette is not None:
            packed = pack_indices(r["indices_yzx"], len(palette))
        pos = {"x": r["position"][0], "y": r["position"][1], "z": r["position"][2]}
        sz = {"x": r["size"][0], "y": r["size"][1], "z": r["size"][2]}
        region = {
            "Position": pos,
            "Size": sz,
            "BlockStates": packed,
            "TileEntities": r.get("tile_entities", []),
            "Entities": r.get("entities", []),
        }
        if r.get("palette") is not None:
            region["BlockStatePalette"] = palette_entries
        else:
            del region["BlockStates"]  # field-under-test absent, not merely empty
        if "pending_block_ticks" in r:
            region["PendingBlockTicks"] = r["pending_block_ticks"]
        if "pending_fluid_ticks" in r:
            region["PendingFluidTicks"] = r["pending_fluid_ticks"]
        regions_nbt[r["name"]] = region
    root = {
        "MinecraftDataVersion": data_version,
        "Metadata": meta,
        "Regions": regions_nbt,
    }
    if extra_root:
        root.update(extra_root)
    return write_nbt(root, gzip=gzip, root_name="Schematic")
