"""Scene composition: culling, yaw rotation, span computation."""

from typing import NamedTuple

import numpy as np

from litematic.parse.model import Litematic
from litematic.render.errors import RenderError
from litematic.render.pack import Textures

_AIR_NAMES = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}


class _Scene(NamedTuple):
    """Visible blocks after culling and yaw rotation, min-corner at origin."""

    x: np.ndarray  # int32
    y: np.ndarray  # int32
    z: np.ndarray  # int32
    shape: np.ndarray  # uint8
    tex_ok: np.ndarray  # bool
    span_w: int  # canvas width in pixels = 2t * span_w at scale t
    span_h: int  # canvas height in pixels = t * span_h at scale t
    face_slot: np.ndarray  # int32, index into slots; -1 renders magenta
    slots: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]


def _yaw_rot(
    x: np.ndarray, z: np.ndarray, yaw: int, w: int, d: int
) -> tuple[np.ndarray, np.ndarray]:
    if yaw == 1:
        return z, (w - 1) - x
    if yaw == 2:
        return (w - 1) - x, (d - 1) - z
    if yaw == 3:
        return (d - 1) - z, x
    return x, z


def _compose(schem: Litematic, yaw: int, textures: Textures) -> _Scene:
    """Placeholder: the v2 quad compose lands with plan Task 6."""
    raise RenderError(
        "render: quad compose is not wired yet (v2 migration, plan Task 6)"
    )
