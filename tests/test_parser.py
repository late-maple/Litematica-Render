"""parse_schematic/load/loads contract: metadata, regions, envelopes, validation."""

import numpy as np
import pytest
from helpers.litematica_packer import build_litematic_bytes
from helpers.nbt_writer import write_nbt

from litematic import LitematicError, LitematicFormatError
from litematic.model import BlockState, Vec3i
from litematic.parser import load, loads

PALETTE = [
    ("minecraft:air", {}),
    ("minecraft:stone", {}),
    ("minecraft:oak_stairs", {"facing": "north"}),
]


def _rng_indices(size, n_palette=3, seed=5):
    rng = np.random.default_rng(seed)
    n = abs(size[0] * size[1] * size[2])
    return rng.integers(0, n_palette, size=n).astype(np.uint16)


def _region(name="Main", size=(2, 3, 4), **kwargs):
    r = {
        "name": name,
        "position": (0, 0, 0),
        "size": size,
        "palette": PALETTE,
        "indices_yzx": _rng_indices(tuple(abs(s) for s in size)),
    }
    r.update(kwargs)
    return r


def test_two_regions_metadata_and_blocks():
    data = build_litematic_bytes(
        [_region("Main", size=(2, 3, 4)), _region("Annex", size=(1, 1, 1))],
        metadata_overrides={
            "Name": "My Build",
            "Description": "desc",
            "Author": "late_maple",
            "EnclosingSize": {"x": 10, "y": 20, "z": 30},
            "TotalBlocks": 42,
            "TotalVolume": 100,
            "TimeCreated": 1234,
            "TimeModified": 5678,
        },
        data_version=3337,
    )
    schem = loads(data)
    assert schem.metadata.name == "My Build"
    assert schem.metadata.author == "late_maple"
    assert schem.metadata.description == "desc"
    assert schem.metadata.enclosing_size == Vec3i(10, 20, 30)
    assert schem.metadata.total_blocks == 42
    assert schem.metadata.total_volume == 100
    assert schem.metadata.time_created == 1234
    assert schem.metadata.time_modified == 5678
    assert schem.metadata.data_version == 3337
    assert set(schem.regions) == {"Main", "Annex"}

    region = schem.regions["Main"]
    assert region.size == Vec3i(2, 3, 4)
    assert region.position == Vec3i(0, 0, 0)
    assert region.palette[2] == BlockState("minecraft:oak_stairs", {"facing": "north"})
    assert not region.is_unpacked
    vals = region.block_indices
    assert region.is_unpacked
    assert vals.shape == (2, 3, 4)
    indices = _rng_indices((2, 3, 4))
    for x in range(2):
        for y in range(3):
            for z in range(4):
                assert vals[x, y, z] == indices[(y * 4 + z) * 2 + x]


def test_missing_optional_metadata_defaults():
    meta = {"Name": "OnlyName"}
    # EnclosingSize/Description/Author and counters all absent
    data = build_litematic_bytes([_region()], metadata_overrides=meta)
    schem = loads(data)
    assert schem.metadata.name == "OnlyName"
    assert schem.metadata.author == ""
    assert schem.metadata.description == ""
    assert schem.metadata.enclosing_size == Vec3i(0, 0, 0)
    assert schem.metadata.total_blocks == 0
    assert schem.metadata.data_version == 3337


def test_negative_size_normalizes():
    data = build_litematic_bytes([_region(size=(-5, 3, 4), position=(10, 0, 0))])
    region = loads(data).regions["Main"]
    assert region.size == Vec3i(5, 3, 4)
    assert region.position == Vec3i(5, 0, 0)


def test_missing_palette_raises_with_region_name():
    data = build_litematic_bytes([_region(palette=None)])
    with pytest.raises(LitematicFormatError, match="Main"):
        loads(data)


def test_block_states_length_mismatch_raises_at_load():
    packed = np.zeros(99, dtype=np.int64)  # 12 blocks * 2 bits -> expects 1 long
    data = build_litematic_bytes([_region(size=(3, 2, 2), packed=packed)])
    with pytest.raises(LitematicFormatError, match=r"Main.*99"):
        loads(data)


def test_empty_regions_raises():
    data = build_litematic_bytes([])
    with pytest.raises(LitematicError):
        loads(data)


def test_out_of_range_index_raises_on_access_with_region_name():
    # 2x2x2 -> 8 blocks; palette has 3 entries (valid indices 0..2); the
    # crafted longs make every 2-bit entry decode to 3, one past the limit
    crafted = np.array([0xFFFF], dtype=np.int64)
    data = build_litematic_bytes([_region(size=(2, 2, 2), packed=crafted)])
    region = loads(data).regions["Main"]
    with pytest.raises(LitematicFormatError, match="Main"):
        region.block_indices


def test_tile_entities_and_entities_envelopes():
    te = {"x": 1, "y": 2, "z": 3, "id": "minecraft:chest", "Custom": "stuff"}
    ent = {"id": "minecraft:pig", "Pos": [1.5, 2.0, 3.5], "Health": 10.0}
    data = build_litematic_bytes([_region(tile_entities=[te], entities=[ent])])
    region = loads(data).regions["Main"]
    assert len(region.block_entities) == 1
    be = region.block_entities[0]
    assert be.pos == (1, 2, 3)
    assert be.id == "minecraft:chest"
    assert be.data["Custom"] == "stuff"
    assert len(region.entities) == 1
    e = region.entities[0]
    assert e.id == "minecraft:pig"
    assert e.pos == (1.5, 2.0, 3.5)
    assert e.data["Health"] == 10.0


def test_pending_fields_kept_verbatim():
    data = build_litematic_bytes(
        [_region(pending_states={"Pending": 1}, pending_counts={"Count": 2})]
    )
    region = loads(data).regions["Main"]
    assert region.pending_states == {"Pending": 1}
    assert region.pending_counts == {"Count": 2}


def test_absent_pending_fields_are_none():
    data = build_litematic_bytes([_region()])
    region = loads(data).regions["Main"]
    assert region.pending_states is None
    assert region.pending_counts is None


def test_unknown_root_fields_ignored():
    data = build_litematic_bytes([_region()], extra_root={"FutureField": {"a": 1}})
    assert loads(data).regions["Main"].size == Vec3i(2, 3, 4)


def test_load_from_path(tmp_path):
    data = build_litematic_bytes([_region()], metadata_overrides={"Name": "PathLoaded"})
    target = tmp_path / "schem.litematic"
    target.write_bytes(data)
    schem = load(target)
    assert schem.regions["Main"].size == Vec3i(2, 3, 4)
    schem2 = load(str(target))
    assert schem2.metadata.name == "PathLoaded"


def test_load_missing_file_raises_litematic_error(tmp_path):
    with pytest.raises(LitematicError):
        load(tmp_path / "nope.litematic")


# --- malformed-input containment: native exceptions must not escape ---

REGION_NBT = {
    "Position": {"x": 0, "y": 0, "z": 0},
    "Size": {"x": 2, "y": 2, "z": 2},
    "BlockStatePalette": [{"Name": "minecraft:air"}],
    "BlockStates": np.zeros(1, dtype=np.int64),
}


def _load_root(regions=None, metadata=None):
    root: dict = {"DataVersion": 3337}
    root["Regions"] = regions if regions is not None else {"Main": dict(REGION_NBT)}
    if metadata is not None:
        root["Metadata"] = metadata
    return loads(write_nbt(root))


def _region_with(**overrides):
    return {"Main": dict(REGION_NBT, **overrides)}


@pytest.mark.parametrize(
    "metadata",
    [
        [1, 2],  # not a compound
        {"EnclosingSize": {"x": 1, "y": 2}},  # missing z
        {"TotalBlocks": "abc"},  # non-numeric counter
        {"TimeCreated": "1.5"},  # non-integer timestamp
    ],
)
def test_malformed_metadata_raises(metadata):
    with pytest.raises(LitematicFormatError):
        _load_root(metadata=metadata)


@pytest.mark.parametrize(
    "palette",
    [
        {"A": {"Name": "x"}},  # compound instead of list
        [{"Properties": {}}],  # entry missing Name
        [{"Name": "x", "Properties": [1]}],  # Properties not a compound
    ],
)
def test_malformed_palette_raises(palette):
    with pytest.raises(LitematicFormatError):
        _load_root(regions=_region_with(BlockStatePalette=palette))


@pytest.mark.parametrize(
    "tile_entities",
    [
        {"0": {"x": 1}},  # compound instead of list
        [{"x": 1, "y": 2, "id": "minecraft:chest"}],  # missing z
        ["not-a-compound"],
    ],
)
def test_malformed_tile_entities_raises(tile_entities):
    with pytest.raises(LitematicFormatError):
        _load_root(regions=_region_with(TileEntities=tile_entities))


@pytest.mark.parametrize(
    "entities",
    [
        [{"id": "minecraft:pig"}],  # missing pos
        [{"id": "minecraft:pig", "Pos": [1.0, 2.0]}],  # pos too short
        ["not-a-compound"],
    ],
)
def test_malformed_entities_raises(entities):
    with pytest.raises(LitematicFormatError):
        _load_root(regions=_region_with(Entities=entities))


def test_block_states_wrong_dtype_fails_at_load():
    # element count matches the length check, dtype does not
    with pytest.raises(LitematicFormatError):
        _load_root(regions=_region_with(BlockStates=np.zeros(1, dtype=np.int32)))


def test_regions_wrong_shape_raises():
    with pytest.raises(LitematicFormatError):
        _load_root(regions=[{"name": "Main"}])


def test_region_value_wrong_shape_raises():
    with pytest.raises(LitematicFormatError):
        _load_root(regions={"Main": [1, 2]})


def test_string_coordinates_raise():
    with pytest.raises(LitematicFormatError):
        _load_root(
            regions=_region_with(Position={"x": "1.5", "y": 0, "z": 0})
        )
