"""Read-only data model: metadata, block states, and lazily materialized regions."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, NamedTuple, Optional, Sequence

import numpy as np


class Vec3i(NamedTuple):
    """A 3D integer vector (x, y, z)."""

    x: int
    y: int
    z: int


@dataclass(frozen=True)
class BlockState:
    """A palette entry: block id plus blockstate properties."""

    name: str
    properties: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "properties", MappingProxyType(dict(self.properties))
        )

    def __hash__(self) -> int:
        return hash((self.name, frozenset(self.properties.items())))


@dataclass(frozen=True)
class LitematicMetadata:
    """Schematic-level metadata from the Metadata compound."""

    name: str
    author: str
    description: str
    enclosing_size: Vec3i
    total_blocks: int
    total_volume: int
    time_created: int
    time_modified: int
    data_version: int


@dataclass(frozen=True)
class BlockEntity:
    """A tile entity: position, id, and its raw NBT payload as a plain mapping.

    The mapping is read-only at the top level only; nested compounds
    (a chest's items, for example) stay mutable dicts.
    """

    pos: tuple[int, int, int]
    id: str
    data: Mapping[str, object]


@dataclass(frozen=True)
class Entity:
    """An entity: id, position, and its raw NBT payload as a plain mapping.

    The mapping is read-only at the top level only; nested compounds
    stay mutable dicts.
    """

    id: str
    pos: tuple[float, float, float]
    data: Mapping[str, object]


@dataclass(frozen=True)
class Litematic:
    """A parsed schematic: metadata plus named regions.

    `regions` is exposed as a read-only mapping.
    """

    metadata: LitematicMetadata
    regions: Mapping[str, "Region"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", MappingProxyType(self.regions))


class Region:
    """One schematic region.

    Block data is kept in packed form; `block_indices` materializes it on
    first access and caches the result until `release()`.
    `pending_block_ticks`/`pending_fluid_ticks` are sequences of scheduled-tick
    compounds ({x, y, z, Time, Priority, Block or Fluid, SubTick}), retained
    verbatim (semantics defined by the writing Litematica version).
    Not thread-safe.
    """

    def __init__(
        self,
        name: str,
        position: Vec3i,
        size: Vec3i,
        palette: Sequence[BlockState],
        block_entities: tuple[BlockEntity, ...],
        entities: tuple[Entity, ...],
        pending_block_ticks: Optional[tuple[Mapping[str, object], ...]],
        pending_fluid_ticks: Optional[tuple[Mapping[str, object], ...]],
        packed_indices: np.ndarray,
        _unpack: Callable[[], np.ndarray],
        _unpack_slab: Optional[Callable[[int, int], np.ndarray]] = None,
    ) -> None:
        self._name = name
        self._position = position
        self._size = size
        self._palette = tuple(palette)
        self._block_entities = block_entities
        self._entities = entities
        self._pending_block_ticks = pending_block_ticks
        self._pending_fluid_ticks = pending_fluid_ticks
        # Read-only view: the caller keeps ownership of the array it passed.
        packed = packed_indices.view()
        packed.flags.writeable = False
        self._packed_indices = packed
        self._unpack = _unpack
        self._unpack_slab = _unpack_slab
        self._indices: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        """Region name as written in the file."""
        return self._name

    @property
    def position(self) -> Vec3i:
        """Min corner in schematic coordinates (negative sizes normalized)."""
        return self._position

    @property
    def size(self) -> Vec3i:
        """Region extents along x, y, z; always positive."""
        return self._size

    @property
    def palette(self) -> tuple[BlockState, ...]:
        """Block states addressable by the values in `block_indices`."""
        return self._palette

    @property
    def block_entities(self) -> tuple[BlockEntity, ...]:
        """Tile entities (chests, signs, spawners, ...) in this region."""
        return self._block_entities

    @property
    def entities(self) -> tuple[Entity, ...]:
        """Entities (mobs, items, item frames, ...) in this region."""
        return self._entities

    @property
    def pending_block_ticks(self) -> Optional[tuple[Mapping[str, object], ...]]:
        """Scheduled block ticks, one compound per tick; None if absent."""
        return self._pending_block_ticks

    @property
    def pending_fluid_ticks(self) -> Optional[tuple[Mapping[str, object], ...]]:
        """Scheduled fluid ticks, one compound per tick; None if absent."""
        return self._pending_fluid_ticks

    @property
    def packed_indices(self) -> np.ndarray:
        """The raw packed BlockStates long array, read-only."""
        return self._packed_indices

    @property
    def is_unpacked(self) -> bool:
        """Whether `block_indices` is currently materialized."""
        return self._indices is not None

    @property
    def block_indices(self) -> np.ndarray:
        """Palette indices as a read-only (x, y, z) array; materializes lazily."""
        if self._indices is None:
            indices = self._unpack()
            indices.flags.writeable = False
            self._indices = indices
        return self._indices

    def slab_indices(self, y0: int, y1: int) -> np.ndarray:
        """Palette indices for rows [y0, y1), shape (x, y1-y0, z); not cached.

        Unpacks only the slab's contiguous entry span of the packed stream
        when the parser injected a slab unpacker, so callers can stream a
        region without ever materializing `block_indices`.
        """
        if not (0 <= y0 < y1 <= self._size.y):
            raise IndexError(
                f"Slab rows out of range: [{y0}, {y1}) for size {self._size}"
            )
        if self._unpack_slab is not None:
            arr = self._unpack_slab(y0, y1)
        else:
            arr = np.ascontiguousarray(self.block_indices[:, y0:y1, :])
        arr.flags.writeable = False
        return arr

    def get_block(self, x: int, y: int, z: int) -> BlockState:
        """Return the block state at local coordinates (x, y, z).

        Out-of-range coordinates are a caller bug and raise IndexError;
        file corruption raises LitematicFormatError on first unpack.
        """
        size_x, size_y, size_z = self._size
        if not (0 <= x < size_x and 0 <= y < size_y and 0 <= z < size_z):
            raise IndexError(
                f"Block coordinates out of range: ({x}, {y}, {z})"
                f" for size {self._size}"
            )
        index = int(self.block_indices[x, y, z])
        return self._palette[index]

    def release(self) -> None:
        """Drop the materialized `block_indices` cache to reclaim memory."""
        self._indices = None
