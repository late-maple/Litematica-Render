"""Data model contract: value types and lazy region materialization."""

import numpy as np
import pytest

from litematic.parse.model import (
    BlockState,
    Litematic,
    LitematicMetadata,
    Region,
    Vec3i,
)


def test_block_state_value_equality_and_hash():
    a = BlockState("minecraft:stone", {})
    b = BlockState("minecraft:stone", {})
    assert a == b
    assert len({a, b}) == 1
    stairs_north = BlockState("minecraft:oak_stairs", {"facing": "north"})
    stairs_south = BlockState("minecraft:oak_stairs", {"facing": "south"})
    assert stairs_north != stairs_south
    assert stairs_north == BlockState("minecraft:oak_stairs", {"facing": "north"})
    assert stairs_north != a


def test_block_state_properties_read_only():
    props = BlockState("minecraft:x", {"a": "b"}).properties
    with pytest.raises(TypeError):
        props["a"] = "c"


def _make_region(
    unpack, size=Vec3i(2, 3, 4), palette=None, packed=None
):
    palette = palette or [
        BlockState("minecraft:air", {}),
        BlockState("minecraft:stone", {}),
    ]
    return Region(
        name="Main",
        position=Vec3i(0, 0, 0),
        size=size,
        palette=palette,
        block_entities=(),
        entities=(),
        pending_block_ticks=None,
        pending_fluid_ticks=None,
        packed_indices=(
            packed if packed is not None else np.zeros(8, dtype=np.int64)
        ),
        _unpack=unpack,
    )


def test_block_indices_lazy_and_cached():
    calls = []
    arr = np.zeros((2, 3, 4), dtype=np.uint16)

    def unpack():
        calls.append(1)
        return arr

    region = _make_region(unpack)
    assert not region.is_unpacked
    first = region.block_indices
    second = region.block_indices
    assert region.is_unpacked
    assert first is second
    assert len(calls) == 1


def test_release_drops_cache_and_rematerializes():
    calls = []
    arr = np.zeros((2, 3, 4), dtype=np.uint16)

    def unpack():
        calls.append(1)
        return arr

    region = _make_region(unpack)
    region.block_indices
    region.release()
    assert not region.is_unpacked
    region.block_indices
    assert len(calls) == 2


def test_get_block_and_bounds():
    arr = np.ones((2, 3, 4), dtype=np.uint16)
    arr[0, 0, 0] = 0
    region = _make_region(lambda: arr)
    assert region.get_block(0, 0, 0).name == "minecraft:air"
    assert region.get_block(1, 2, 3).name == "minecraft:stone"
    with pytest.raises(IndexError):
        region.get_block(2, 0, 0)
    with pytest.raises(IndexError):
        region.get_block(-1, 0, 0)
    with pytest.raises(IndexError):
        region.get_block(0, 3, 0)


def test_palette_is_immutable_snapshot():
    source = [
        BlockState("minecraft:air", {}),
        BlockState("minecraft:stone", {}),
    ]
    region = _make_region(
        lambda: np.zeros((2, 3, 4), dtype=np.uint16), palette=source
    )
    source.append(BlockState("minecraft:tnt", {}))
    assert len(region.palette) == 2
    with pytest.raises(AttributeError):
        region.palette.append(BlockState("minecraft:tnt", {}))


def test_indices_and_packed_read_only():
    arr = np.zeros((2, 3, 4), dtype=np.uint16)
    region = _make_region(lambda: arr)
    with pytest.raises(ValueError):
        region.block_indices[0, 0, 0] = 1
    with pytest.raises(ValueError):
        region.packed_indices[0] = 1


def test_packed_read_only_view_does_not_touch_caller_array():
    packed = np.zeros(8, dtype=np.int64)
    region = _make_region(
        lambda: np.zeros((2, 3, 4), dtype=np.uint16), packed=packed
    )
    assert packed.flags.writeable
    assert not region.packed_indices.flags.writeable
    assert region.packed_indices.base is not None


def test_regions_mapping_read_only():
    region = _make_region(lambda: np.zeros((2, 3, 4), dtype=np.uint16))
    meta = LitematicMetadata("", "", "", Vec3i(0, 0, 0), 0, 0, 0, 0, 0, 0)
    schem = Litematic(metadata=meta, regions={"Main": region})
    with pytest.raises(TypeError):
        schem.regions["other"] = region


def test_region_members_documented():
    for member in (
        "get_block",
        "release",
        "block_indices",
        "is_unpacked",
        "packed_indices",
    ):
        assert getattr(Region, member).__doc__, member
