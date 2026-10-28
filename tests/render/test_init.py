"""Render subpackage scaffold: public error type, version bump, top-level isolation."""

import litematic
import litematic.render

_TOP_LEVEL_ALL = [
    "BlockEntity",
    "BlockState",
    "Entity",
    "Litematic",
    "LitematicError",
    "LitematicFormatError",
    "LitematicMetadata",
    "NbtError",
    "Region",
    "Vec3i",
    "load",
    "loads",
]


def test_render_error_subclasses_litematic_error():
    assert issubclass(litematic.render.RenderError, litematic.LitematicError)


def test_version_is_0_3_0():
    assert litematic.__version__ == "0.3.0"


def test_top_level_all_unchanged_by_render():
    assert litematic.__all__ == _TOP_LEVEL_ALL
