"""Offline isometric rendering of parsed .litematic schematics."""

from pathlib import Path

import numpy as np

from litematic.parse.model import Litematic
from litematic.render.errors import RenderError
from litematic.render.pack import load_pack
from litematic.render.png import encode_png
from litematic.render.scene.isometric import _AIR_NAMES, _compose
from litematic.render.scene.painter import _choose_scale, _downsample, _paint

__all__ = [
    "RenderError",
    "isometric",
    "isometric_png",
    "to_png",
    "unresolved_colors",
]

_MAX_SIZE_BOUNDS = (16, 8192)
_SS_BOUNDS = (1, 4)


def isometric(
    schem: Litematic,
    *,
    textures_path: str | Path | None = None,
    yaw: int = 0,
    max_size: int = 2048,
    supersample: int = 2,
    background: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Render a schematic to an RGBA isometric image within max_size."""
    if yaw not in (0, 1, 2, 3):
        raise RenderError(f"render: yaw must be 0-3, got {yaw}")
    if not _MAX_SIZE_BOUNDS[0] <= max_size <= _MAX_SIZE_BOUNDS[1]:
        raise RenderError(
            f"render: max_size must be "
            f"{_MAX_SIZE_BOUNDS[0]}-{_MAX_SIZE_BOUNDS[1]}, got {max_size}"
        )
    if not _SS_BOUNDS[0] <= supersample <= _SS_BOUNDS[1]:
        raise RenderError(
            f"render: supersample must be {_SS_BOUNDS[0]}-{_SS_BOUNDS[1]}, "
            f"got {supersample}"
        )
    bg = _background(background)
    textures = load_pack(_require_pack(textures_path))
    scene = _compose(schem, yaw, textures)
    if len(scene.x) == 0:
        raise RenderError("render: no blocks to render")
    t, _ = _choose_scale(scene.span_w, scene.span_h, max_size, supersample)
    img = _paint(scene, t, bg)
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        factor = max_size / max(height, width)
        img = _downsample(
            img, max(1, round(width * factor)), max(1, round(height * factor))
        )
    return img


def isometric_png(
    schem: Litematic,
    *,
    textures_path: str | Path | None = None,
    yaw: int = 0,
    max_size: int = 2048,
    supersample: int = 2,
    background: tuple[int, int, int, int] | None = None,
) -> bytes:
    """Render a schematic and encode the image as PNG bytes."""
    return encode_png(
        isometric(
            schem,
            textures_path=textures_path,
            yaw=yaw,
            max_size=max_size,
            supersample=supersample,
            background=background,
        )
    )


def to_png(image: np.ndarray) -> bytes:
    """Encode an RGBA uint8 image array as PNG bytes."""
    return encode_png(image)


def unresolved_colors(
    schem: Litematic, *, textures_path: str | Path
) -> set[str]:
    """Return block names the pack cannot cover; these render magenta."""
    textures = load_pack(_require_pack(textures_path))
    return {
        state.name
        for region in schem.regions.values()
        for state in region.palette
        if state.name not in _AIR_NAMES
        and textures.resolve(state).miss is not None
    }


def _background(
    value: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    if value is None:
        return (0, 0, 0, 0)
    if (
        isinstance(value, tuple)
        and len(value) == 4
        and all(
            isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255
            for v in value
        )
    ):
        return value
    raise RenderError("render: background must be None or 4 ints in [0, 255]")


def _require_pack(textures_path: str | Path | None) -> str | Path:
    if textures_path is None:
        raise RenderError(
            "render: a resource pack zip is required (pass textures_path)"
        )
    return textures_path
