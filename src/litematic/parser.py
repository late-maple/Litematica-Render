"""Litematica palette-index bit unpacking and file-tree parsing."""

from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, cast

import numpy as np

from litematic.errors import LitematicError, LitematicFormatError
from litematic.model import (
    BlockEntity,
    BlockState,
    Entity,
    Litematic,
    LitematicMetadata,
    Region,
    Vec3i,
)
from litematic.nbt import read_nbt


def _bits_for(palette_size: int) -> int:
    """Bit width per entry: enough for palette_size values, at least 2."""
    return max(2, (palette_size - 1).bit_length())


def _unpack_indices(packed: np.ndarray, bits: int, size: Vec3i) -> np.ndarray:
    """Unpack litematica's continuous bit stream into an (x, y, z) index array.

    Entry i starts at bit i*bits of the long array, filling each long upward
    from its low bit; an entry straddling a boundary continues into the low
    bits of the next long. File order is
    YZX: index = (y*lenZ + z)*lenX + x. Palettes wider than 16 bits need
    uint32 output. Entries are processed in fixed-size chunks so transient
    memory stays a small multiple of the output array regardless of size.
    """
    count = size.x * size.y * size.z
    words = packed.view(np.uint64)  # logical shifts; int64 storage is untouched
    dtype = np.uint16 if bits <= 16 else np.uint32
    out = np.empty(count, dtype=dtype)
    mask = np.uint64((1 << bits) - 1)
    chunk = 1 << 18
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        positions = np.arange(start, stop, dtype=np.uint64) * np.uint64(bits)
        long_index = positions >> np.uint64(6)
        bit_offset = positions & np.uint64(63)
        values = words[long_index] >> bit_offset
        spill = bit_offset.astype(np.int64) + bits - 64
        crossing = np.nonzero(spill > 0)[0]
        if crossing.size:
            values[crossing] |= words[long_index[crossing] + 1] << (
                np.uint64(64) - bit_offset[crossing]
            )
        values &= mask
        out[start:stop] = values
    return out.reshape(size.y, size.z, size.x).transpose(2, 0, 1)


def _vec3(compound: Mapping, field: str, source: str) -> Vec3i:
    try:
        c = compound[field]
        return Vec3i(int(c["x"]), int(c["y"]), int(c["z"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise LitematicFormatError(f"{source}: missing or malformed {field}") from exc


def _string(value: object, what: str, where: str) -> str:
    if not isinstance(value, str):
        raise LitematicFormatError(f"{where}: {what} is not a string")
    return value


def _parse_metadata(root: Mapping, source: str) -> LitematicMetadata:
    try:
        meta = root.get("Metadata") or {}
        if not isinstance(meta, dict):
            raise TypeError("Metadata is not a compound")
        where = f"{source}: Metadata"
        size = meta.get("EnclosingSize")
        if size is None:
            enclosing = Vec3i(0, 0, 0)
        else:
            enclosing = Vec3i(int(size["x"]), int(size["y"]), int(size["z"]))
        return LitematicMetadata(
            name=_string(meta.get("Name", ""), "Name", where),
            author=_string(meta.get("Author", ""), "Author", where),
            description=_string(meta.get("Description", ""), "Description", where),
            enclosing_size=enclosing,
            total_blocks=int(meta.get("TotalBlocks", 0)),
            total_volume=int(meta.get("TotalVolume", 0)),
            time_created=int(meta.get("TimeCreated", 0)),
            time_modified=int(meta.get("TimeModified", 0)),
            data_version=int(root.get("DataVersion", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LitematicFormatError(f"{source}: malformed Metadata: {exc}") from exc


def _parse_palette(entries: object, source: str) -> list[BlockState]:
    if not isinstance(entries, list) or not entries:
        raise LitematicFormatError(f"{source}: BlockStatePalette is missing or empty")
    palette = []
    try:
        for entry in entries:
            props = entry.get("Properties") or {}
            palette.append(
                BlockState(_string(entry["Name"], "Name", source), dict(props))
            )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise LitematicFormatError(f"{source}: malformed palette entry: {exc}") from exc
    return palette


def _make_unpacker(
    packed: np.ndarray,
    bits: int,
    size: Vec3i,
    source: str,
    palette_size: int,
) -> Callable[[], np.ndarray]:
    def unpack() -> np.ndarray:
        arr = _unpack_indices(packed, bits, size)
        if arr.size and int(arr.max()) >= palette_size:
            raise LitematicFormatError(
                f"{source}: block index {int(arr.max())} out of range "
                f"for palette of {palette_size}"
            )
        return arr

    return unpack


def _read_only(value: object) -> object:
    """Dicts become read-only mappings; other payloads pass through verbatim."""
    return MappingProxyType(dict(value)) if isinstance(value, dict) else value


def _parse_block_entities(entries: object, where: str) -> tuple[BlockEntity, ...]:
    if not isinstance(entries, list):
        raise LitematicFormatError(f"{where}: TileEntities is not a list")
    out = []
    for entry in entries:
        try:
            out.append(
                BlockEntity(
                    pos=(int(entry["x"]), int(entry["y"]), int(entry["z"])),
                    id=_string(entry.get("id", ""), "id", where),
                    data=MappingProxyType(dict(entry)),
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise LitematicFormatError(
                f"{where}: malformed tile entity: {exc}"
            ) from exc
    return tuple(out)


def _parse_entities(entries: object, where: str) -> tuple[Entity, ...]:
    if not isinstance(entries, list):
        raise LitematicFormatError(f"{where}: Entities is not a list")
    out = []
    for entry in entries:
        try:
            pos = entry["Pos"]
            out.append(
                Entity(
                    id=_string(entry.get("id", ""), "id", where),
                    pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                    data=MappingProxyType(dict(entry)),
                )
            )
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            IndexError,
        ) as exc:
            raise LitematicFormatError(f"{where}: malformed entity: {exc}") from exc
    return tuple(out)


def _parse_region(name: str, region: Mapping, source: str) -> Region:
    where = f"{source}, region '{name}'"
    try:
        palette = _parse_palette(region["BlockStatePalette"], where)
        packed = region["BlockStates"]
    except KeyError as exc:
        raise LitematicFormatError(f"{where}: missing field {exc}") from exc
    pos_raw = _vec3(region, "Position", where)
    size_raw = _vec3(region, "Size", where)
    size = Vec3i(*(abs(s) for s in size_raw))
    position = Vec3i(
        *(p + (s if s < 0 else 0) for p, s in zip(pos_raw, size_raw))
    )
    if not isinstance(packed, np.ndarray) or packed.dtype != np.int64:
        raise LitematicFormatError(f"{where}: BlockStates is not a long array")
    bits = _bits_for(len(palette))
    volume = size.x * size.y * size.z
    expected_longs = (volume * bits + 63) // 64
    if packed.size != expected_longs:
        raise LitematicFormatError(
            f"{where}: BlockStates has {packed.size} longs, "
            f"expected {expected_longs} ({volume} blocks at {bits} bits)"
        )
    block_entities = _parse_block_entities(region.get("TileEntities", []), where)
    entities = _parse_entities(region.get("Entities", []), where)
    # pending payload shapes are not yet verified against real files;
    # non-mapping payloads may pass through at runtime until they are
    pending_states = cast(
        "Mapping[str, object] | None", _read_only(region.get("PendingBlockStates"))
    )
    pending_counts = cast(
        "Mapping[str, object] | None", _read_only(region.get("PendingBlockCounts"))
    )
    return Region(
        name=name,
        position=position,
        size=size,
        palette=palette,
        block_entities=block_entities,
        entities=entities,
        pending_states=pending_states,
        pending_counts=pending_counts,
        packed_indices=packed,
        _unpack=_make_unpacker(packed, bits, size, where, len(palette)),
    )


def parse_schematic(root: dict, source_name: str = "") -> Litematic:
    """Build a Litematic model from an already-parsed NBT root compound."""
    regions = root.get("Regions")
    if not isinstance(regions, dict) or not regions:
        raise LitematicFormatError(f"{source_name}: no Regions")
    parsed = {}
    for name, raw in regions.items():
        if not isinstance(raw, dict):
            raise LitematicFormatError(
                f"{source_name}, region '{name}': region is not a compound"
            )
        parsed[name] = _parse_region(name, raw, source_name)
    return Litematic(metadata=_parse_metadata(root, source_name), regions=parsed)


def loads(data: bytes) -> Litematic:
    """Parse .litematic bytes into a Litematic model."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise LitematicError(
            f"loads() expects bytes-like input, got {type(data).__name__}"
        )
    return parse_schematic(read_nbt(data), "<bytes>")


def load(path: str | Path) -> Litematic:
    """Read and parse a .litematic file into a Litematic model."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise LitematicError(f"Cannot read {p}: {exc}") from exc
    return parse_schematic(read_nbt(data), str(p))
