"""Read-only data model: metadata, block states, and lazily materialized regions."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, NamedTuple, Optional, Sequence

import numpy as np


class Vec3i(NamedTuple):
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
    """A tile entity: position, id, and its raw NBT payload as a plain mapping."""

    pos: tuple[int, int, int]
    id: str
    data: Mapping[str, object]


@dataclass(frozen=True)
class Entity:
    """An entity: id, position, and its raw NBT payload as a plain mapping."""

    id: str
    pos: tuple[float, float, float]
    data: Mapping[str, object]


@dataclass(frozen=True)
class Litematic:
    metadata: LitematicMetadata
    regions: dict[str, "Region"]


class Region:
    """One schematic region.

    Block data is kept in packed form; `block_indices` materializes it on
    first access and caches the result until `release()`.
    `pending_states`/`pending_counts` are the file's pending-block fields,
    retained verbatim (semantics defined by the writing Litematica version).
    """

    def __init__(
        self,
        name: str,
        position: Vec3i,
        size: Vec3i,
        palette: Sequence[BlockState],
        block_entities: tuple[BlockEntity, ...],
        entities: tuple[Entity, ...],
        pending_states: Optional[Mapping[str, object]],
        pending_counts: Optional[Mapping[str, object]],
        packed_indices: np.ndarray,
        _unpack: Callable[[], np.ndarray],
    ) -> None:
        self._name = name
        self._position = position
        self._size = size
        self._palette = tuple(palette)
        self._block_entities = block_entities
        self._entities = entities
        self._pending_states = pending_states
        self._pending_counts = pending_counts
        self._packed_indices = packed_indices
        self._unpack = _unpack
        self._indices: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self) -> Vec3i:
        return self._position

    @property
    def size(self) -> Vec3i:
        return self._size

    @property
    def palette(self) -> tuple[BlockState, ...]:
        return self._palette

    @property
    def block_entities(self) -> tuple[BlockEntity, ...]:
        return self._block_entities

    @property
    def entities(self) -> tuple[Entity, ...]:
        return self._entities

    @property
    def pending_states(self) -> Optional[Mapping[str, object]]:
        return self._pending_states

    @property
    def pending_counts(self) -> Optional[Mapping[str, object]]:
        return self._pending_counts

    @property
    def packed_indices(self) -> np.ndarray:
        return self._packed_indices

    @property
    def is_unpacked(self) -> bool:
        return self._indices is not None

    @property
    def block_indices(self) -> np.ndarray:
        if self._indices is None:
            self._indices = self._unpack()
        return self._indices

    def get_block(self, x: int, y: int, z: int) -> BlockState:
        size_x, size_y, size_z = self._size
        if not (0 <= x < size_x and 0 <= y < size_y and 0 <= z < size_z):
            raise IndexError(
                f"Block coordinates out of range: ({x}, {y}, {z})"
                f" for size {self._size}"
            )
        index = int(self.block_indices[x, y, z])
        return self._palette[index]

    def release(self) -> None:
        self._indices = None
