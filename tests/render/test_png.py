"""PNG encoder contract: Pillow roundtrip, zeroing, determinism, validation."""

import io

import numpy as np
import pytest
from PIL import Image

from litematic.render import RenderError
from litematic.render.png import encode_png


def _decode(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def test_roundtrip_random_image():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (7, 5, 4), dtype=np.uint8)
    image[..., 3] = rng.integers(1, 256, (7, 5))  # keep every pixel visible
    assert np.array_equal(_decode(encode_png(image)), image)


def test_transparent_pixels_rgb_zeroed():
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[..., :3] = 200  # garbage color under full transparency
    decoded = _decode(encode_png(image))
    assert (decoded[..., 3] == 0).all()
    assert (decoded[..., :3] == 0).all()


def test_encoding_is_deterministic():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (3, 4, 4), dtype=np.uint8)
    assert encode_png(image) == encode_png(image)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4, 3), dtype=np.uint8),  # missing alpha channel
        np.zeros((4, 4, 4)),  # wrong dtype (float64)
        np.zeros(16, dtype=np.uint8),  # not an image at all
    ],
)
def test_rejects_invalid_input(bad):
    with pytest.raises(RenderError, match="encode:"):
        encode_png(bad)
