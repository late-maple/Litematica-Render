"""Minimal Java NBT reader: streaming gzip, big-endian, arrays decoded to numpy."""

import gzip
import io
import struct
import zlib
from typing import Protocol

import numpy as np

from litematic.parse.errors import NbtError

_GZIP_MAGIC = b"\x1f\x8b"

# 256 levels is far beyond any real schematic and costs 2 stack frames per
# level, staying safely under CPython's default recursion limit of 1000
_MAX_DEPTH = 256

_TAG_END = 0
_TAG_BYTE = 1
_TAG_SHORT = 2
_TAG_INT = 3
_TAG_LONG = 4
_TAG_FLOAT = 5
_TAG_DOUBLE = 6
_TAG_BYTE_ARRAY = 7
_TAG_STRING = 8
_TAG_LIST = 9
_TAG_COMPOUND = 10
_TAG_INT_ARRAY = 11
_TAG_LONG_ARRAY = 12

_ARRAY_DTYPES = {
    _TAG_BYTE_ARRAY: (">i1", 1, np.int8),
    _TAG_INT_ARRAY: (">i4", 4, np.int32),
    _TAG_LONG_ARRAY: (">i8", 8, np.int64),
}

_SCALARS = {
    _TAG_BYTE: (">b", 1),
    _TAG_SHORT: (">h", 2),
    _TAG_INT: (">i", 4),
    _TAG_LONG: (">q", 8),
    _TAG_FLOAT: (">f", 4),
    _TAG_DOUBLE: (">d", 8),
}


def _decode_java_utf8(data: bytes) -> str:
    """Decode Java's modified UTF-8 (DataOutput.writeUTF): U+0000 as the
    overlong sequence C0 80, supplementary characters as CESU-8 surrogate pairs."""
    if b"\xc0\x80" in data:
        data = data.replace(b"\xc0\x80", b"\x00")
    try:
        text = data.decode("utf-8", errors="surrogatepass")
        if any("\ud800" <= ch <= "\udfff" for ch in text):
            text = text.encode("utf-16", "surrogatepass").decode("utf-16")
        return text
    except UnicodeDecodeError as exc:
        raise NbtError(f"Invalid UTF-8 in NBT string: {exc}") from exc


class _Readable(Protocol):
    """The only stream capability the reader relies on."""

    def read(self, size: int = -1) -> bytes: ...


class _Reader:
    """Reads an NBT stream sequentially. Any short read raises NbtError."""

    __slots__ = ("_stream",)

    def __init__(self, stream: _Readable) -> None:
        self._stream = stream

    def read_root(self) -> dict:
        root_tag = self.u8()
        if root_tag != _TAG_COMPOUND:
            raise NbtError(f"Root tag must be TAG_Compound, got type {root_tag}")
        self.string()  # root tag name is discarded
        return self._compound(0)

    def _read(self, size: int) -> bytes:
        data = self._stream.read(size)
        if len(data) != size:
            raise NbtError(
                f"NBT stream truncated: needed {size} bytes, got {len(data)}"
            )
        return data

    def u8(self) -> int:
        return self._read(1)[0]

    def string(self) -> str:
        length = struct.unpack(">H", self._read(2))[0]
        return _decode_java_utf8(self._read(length))

    def payload(self, tag: int, depth: int = 0) -> object:
        scalar = _SCALARS.get(tag)
        if scalar is not None:
            fmt, size = scalar
            return struct.unpack(fmt, self._read(size))[0]
        if tag == _TAG_STRING:
            return self.string()
        array = _ARRAY_DTYPES.get(tag)
        if array is not None:
            return self._array(*array)
        if tag == _TAG_LIST:
            return self._list(depth)
        if tag == _TAG_COMPOUND:
            return self._compound(depth)
        raise NbtError(f"Unknown tag type {tag}")

    def _array(self, endian_dtype: str, item_size: int, native: type) -> np.ndarray:
        length = self._i32()
        if length < 0:
            raise NbtError(f"Negative array length: {length}")
        buf = self._read(length * item_size)
        return np.frombuffer(buf, dtype=endian_dtype).astype(native)

    def _list(self, depth: int) -> list:
        self._check_depth(depth)
        element_tag = self.u8()
        length = self._i32()
        if length < 0:
            raise NbtError(f"Negative list length: {length}")
        if element_tag == _TAG_END:
            if length:
                raise NbtError("TAG_List element type is End but length is non-zero")
            return []
        return [self.payload(element_tag, depth + 1) for _ in range(length)]

    def _compound(self, depth: int) -> dict:
        self._check_depth(depth)
        out: dict = {}
        while True:
            tag = self.u8()
            if tag == _TAG_END:
                return out
            name = self.string()
            out[name] = self.payload(tag, depth + 1)

    def _check_depth(self, depth: int) -> None:
        if depth >= _MAX_DEPTH:
            raise NbtError(f"NBT nesting deeper than {_MAX_DEPTH} levels")

    def _i32(self) -> int:
        return struct.unpack(">i", self._read(4))[0]


def read_nbt(data: bytes) -> dict:
    """Parse gzip-compressed or raw Java NBT and return the root compound payload.

    Compound -> dict, List -> list, string -> str, scalars -> int/float;
    the three array kinds -> native-endian numpy arrays owning their memory.
    """
    stream: _Readable
    if data[:2] == _GZIP_MAGIC:
        stream = gzip.GzipFile(fileobj=io.BytesIO(data))
    else:
        stream = io.BytesIO(data)
    reader = _Reader(stream)
    try:
        return reader.read_root()
    except (OSError, EOFError, zlib.error) as exc:
        raise NbtError(f"Failed to decompress NBT: {exc}") from exc
