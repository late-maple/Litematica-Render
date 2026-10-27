"""Public API contract: re-exports, __all__ completeness, docstrings."""

import numpy as np
from helpers.litematica_packer import build_litematic_bytes

import litematic


def _schematic_bytes():
    return build_litematic_bytes(
        [
            {
                "name": "Main",
                "position": (0, 0, 0),
                "size": (2, 2, 2),
                "palette": [
                    ("minecraft:air", {}),
                    ("minecraft:stone", {}),
                ],
                "indices_yzx": np.zeros(8, dtype=np.uint16),
            }
        ],
        metadata_overrides={"Name": "ApiCheck"},
    )


def test_loads_end_to_end_via_package():
    schem = litematic.loads(_schematic_bytes())
    assert schem.metadata.name == "ApiCheck"
    region = schem.regions["Main"]
    assert region.get_block(1, 1, 1).name == "minecraft:air"


def test_load_from_path_via_package(tmp_path):
    target = tmp_path / "schem.litematic"
    target.write_bytes(_schematic_bytes())
    schem = litematic.load(target)
    assert schem.regions["Main"].size == litematic.Vec3i(2, 2, 2)


def test_all_names_importable_and_documented():
    assert litematic.__all__, "__all__ must not be empty"
    for name in litematic.__all__:
        obj = getattr(litematic, name)
        assert obj.__doc__, f"{name} is exported but has no docstring"


def test_package_declares_thread_safety_stance():
    assert "thread" in (litematic.__doc__ or "").lower()
