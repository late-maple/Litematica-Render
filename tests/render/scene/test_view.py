"""View axes contract: triad orthonormality, preset equivalence, tiers."""

import numpy as np
import pytest

from litematic.render.errors import RenderError
from litematic.render.quads import (
    TIER_DOWN,
    TIER_EAST,
    TIER_NORTH,
    TIER_SOUTH,
    TIER_UP,
    TIER_WEST,
)
from litematic.render.scene.view import view_axes

_ISO_SLANT = 35.264389682754654  # arctan(1/sqrt(2))


def _unit(v):
    return np.asarray(v, np.float64) / np.linalg.norm(v)


def test_triad_is_orthonormal():
    axes = view_axes(45.0, _ISO_SLANT)
    r, u, t = axes.right, axes.up, axes.toward
    for v in (r, u, t):
        assert np.linalg.norm(v) == pytest.approx(1.0)
    assert float(r @ u) == pytest.approx(0.0, abs=1e-12)
    assert float(r @ t) == pytest.approx(0.0, abs=1e-12)
    assert float(u @ t) == pytest.approx(0.0, abs=1e-12)


def test_legacy_yaw0_preset_faces_up_south_east():
    axes = view_axes(45.0, _ISO_SLANT)
    assert axes.tiers == (TIER_UP, TIER_SOUTH, TIER_EAST)
    assert axes.right == pytest.approx(_unit((1.0, 0.0, -1.0)))
    assert axes.toward == pytest.approx(_unit((1.0, 1.0, 1.0)))


def test_legacy_yaw_presets_rotate_azimuth():
    assert view_axes(135.0, _ISO_SLANT).tiers == (TIER_UP, TIER_SOUTH, TIER_WEST)
    assert view_axes(225.0, _ISO_SLANT).tiers == (TIER_UP, TIER_NORTH, TIER_WEST)
    assert view_axes(315.0, _ISO_SLANT).tiers == (TIER_UP, TIER_NORTH, TIER_EAST)


def test_rotation_wraps_modulo_360():
    a = view_axes(45.0, _ISO_SLANT)
    b = view_axes(405.0, _ISO_SLANT)
    assert a.right == pytest.approx(b.right)
    assert a.toward == pytest.approx(b.toward)
    assert a.tiers == b.tiers


def test_top_down_sees_only_up():
    axes = view_axes(0.0, 90.0)
    assert axes.tiers == (TIER_UP,)
    assert axes.toward == pytest.approx((0.0, 1.0, 0.0))


def test_bottom_up_sees_only_down():
    axes = view_axes(0.0, -90.0)
    assert axes.tiers == (TIER_DOWN,)


def test_horizon_slant_sees_sides_only():
    axes = view_axes(45.0, 0.0)
    assert TIER_UP not in axes.tiers
    assert TIER_SOUTH in axes.tiers and TIER_EAST in axes.tiers


def test_grazing_azimuth_drops_edge_on_face():
    # Camera due south: the east face is edge-on, not visible.
    axes = view_axes(90.0, 0.0)
    assert axes.tiers == (TIER_SOUTH,)


def test_slant_out_of_range_rejected():
    with pytest.raises(RenderError, match="slant"):
        view_axes(0.0, 90.5)
    with pytest.raises(RenderError, match="slant"):
        view_axes(0.0, -90.5)


def test_non_numeric_angles_rejected():
    with pytest.raises(RenderError, match="rotation"):
        view_axes("45", _ISO_SLANT)
    with pytest.raises(RenderError, match="slant"):
        view_axes(45.0, None)
    with pytest.raises(RenderError, match="slant"):
        view_axes(45.0, True)


def test_non_finite_rotation_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(RenderError, match="rotation must be finite"):
            view_axes(bad, 0.0)
        with pytest.raises(RenderError, match="slant must be finite"):
            view_axes(0.0, bad)


def test_int_angles_accepted():
    axes = view_axes(45, 35)
    assert axes.tiers == (TIER_UP, TIER_SOUTH, TIER_EAST)
