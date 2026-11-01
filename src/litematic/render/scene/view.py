"""Free-camera view axes: rotation/slant degrees to an orthonormal triad."""

import math
from typing import NamedTuple

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.quads import TIER_NORMALS

# Pixel scales reproducing the legacy projection at the preset view: a
# unit world step spans COL_SCALE * t screen columns and ROW_SCALE * t
# rows, so the preset equals col = 2t(x-z), row = t(x+z-2y) exactly.
COL_SCALE = 2.0 * 2**0.5
ROW_SCALE = 6.0**0.5
# Slant reproducing the legacy preset: arctan(1/sqrt(2)) = arcsin(1/sqrt(3)).
ISO_SLANT_DEG = 35.264389682754654

_SLANT_BOUNDS = (-90.0, 90.0)


class ViewAxes(NamedTuple):
    """Orthonormal screen triad in world space, plus visible face tiers.

    right points along growing screen columns, up along shrinking rows,
    toward points at the camera; depth away from the camera is -toward.
    """

    right: np.ndarray  # (3,) f64 unit
    up: np.ndarray  # (3,) f64 unit
    toward: np.ndarray  # (3,) f64 unit
    tiers: tuple[int, ...]  # face tiers facing the camera


def view_axes(rotation: float, slant: float) -> ViewAxes:
    """Build the camera triad from view angles in degrees.

    rotation is the camera azimuth wrapped to [0, 360); slant is the
    elevation clamped-valid in [-90, 90]. The legacy yaw presets are
    rotation = 45 + 90 * yaw at slant = arctan(1/sqrt(2)) (~35.264).
    """
    rotation = _as_angle(rotation, "rotation")
    slant = _as_angle(slant, "slant")
    if not _SLANT_BOUNDS[0] <= slant <= _SLANT_BOUNDS[1]:
        raise RenderError(
            f"render: slant must be in [-90, 90], got {slant}"
        )
    theta = np.deg2rad(rotation % 360.0)
    phi = np.deg2rad(slant)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    toward = np.array([cos_p * cos_t, sin_p, cos_p * sin_t])
    right = np.array([sin_t, 0.0, -cos_t])
    up = np.cross(toward, right)
    tiers = tuple(
        tier for tier in range(6) if float(TIER_NORMALS[tier] @ toward) > 1e-9
    )
    return ViewAxes(right, up, toward, tiers)


def project_bounds(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vtx: np.ndarray,
    axes: ViewAxes,
) -> tuple[float, float, float, float]:
    """Exact projected (col_min, col_max, row_min, row_max) at t=1 pixels."""
    wx = x[:, None] + vtx[..., 0] / 16.0
    wy = y[:, None] + vtx[..., 1] / 16.0
    wz = z[:, None] + vtx[..., 2] / 16.0
    col = COL_SCALE * (
        wx * axes.right[0] + wy * axes.right[1] + wz * axes.right[2]
    )
    row = -ROW_SCALE * (
        wx * axes.up[0] + wy * axes.up[1] + wz * axes.up[2]
    )
    return float(col.min()), float(col.max()), float(row.min()), float(row.max())


def _as_angle(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderError(f"render: {name} must be a number, got {value!r}")
    angle = float(value)
    if not math.isfinite(angle):
        raise RenderError(f"render: {name} must be finite, got {value!r}")
    return angle
