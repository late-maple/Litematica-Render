"""Sizing contract: scale-choice fuse and box-filter downsampling."""

import numpy as np
import pytest

from litematic.render.errors import RenderError
from litematic.render.painter import _choose_scale, _downsample


def test_choose_scale_picks_largest_fitting_t():
    assert _choose_scale(10, 12, 2048, 2) == (204, 2)


def test_choose_scale_downgrades_supersample_for_fuse():
    assert _choose_scale(2000, 2000, 8192, 4) == (2, 1)


def test_choose_scale_unsatisfiable_raises():
    with pytest.raises(RenderError):
        _choose_scale(4000, 6000, 2048, 1)


def test_choose_scale_accepts_fuse_boundary():
    assert _choose_scale(2048, 2048, 8192, 1) == (2, 1)


def test_downsample_2x2_to_one_pixel_means():
    img = np.array(
        [
            [[200, 0, 0, 255], [0, 200, 0, 255]],
            [[0, 0, 200, 255], [100, 100, 0, 255]],
        ],
        np.uint8,
    )
    out = _downsample(img, 1, 1)
    assert tuple(out[0, 0]) == (75, 75, 50, 255)


def test_downsample_3_to_2_buckets():
    row = np.array(
        [[[10, 0, 0, 255], [0, 30, 0, 255], [0, 0, 60, 255]]], np.uint8
    )
    out = _downsample(row, 2, 1)
    assert tuple(out[0, 0]) == (5, 15, 0, 255)
    assert tuple(out[0, 1]) == (0, 0, 60, 255)


def test_downsample_alpha_weights_color():
    img = np.array([[[255, 0, 0, 255], [90, 90, 90, 0]]], np.uint8)
    out = _downsample(img, 1, 1)
    assert tuple(out[0, 0]) == (255, 0, 0, 128)


def test_downsample_transparent_stays_zero():
    img = np.zeros((2, 2, 4), np.uint8)
    out = _downsample(img, 1, 1)
    assert tuple(out[0, 0]) == (0, 0, 0, 0)


def test_downsample_identity_returns_copy():
    img = np.array([[[1, 2, 3, 255]]], np.uint8)
    out = _downsample(img, 1, 1)
    assert (out == img).all()
    assert out is not img
