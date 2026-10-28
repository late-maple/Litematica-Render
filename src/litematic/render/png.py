"""Minimal deterministic PNG encoder for RGBA uint8 images."""

import struct
import zlib

import numpy as np

from litematic.render.errors import RenderError

__all__ = ["encode_png"]

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def encode_png(image: np.ndarray) -> bytes:
    """Encode (H, W, 4) uint8 RGBA as PNG; zero transparent RGB; raises RenderError."""
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 4
        or min(image.shape[:2]) < 1
    ):
        raise RenderError(
            "encode: expected (H, W, 4) uint8 array, got "
            f"dtype={getattr(image, 'dtype', type(image).__name__)} "
            f"shape={getattr(image, 'shape', None)}"
        )

    pixels = image.copy()
    transparent = pixels[..., 3] == 0
    pixels[..., :3][transparent] = 0

    height, width = pixels.shape[:2]
    scanlines = np.zeros((height, width * 4 + 1), dtype=np.uint8)
    scanlines[:, 1:] = pixels.reshape(height, width * 4)
    raw = scanlines.tobytes()

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        _SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data))
    )
