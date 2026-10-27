"""Test-only NBT constructor: Python value tree -> NBT bytes."""

import gzip as gzip_module
import struct
from dataclasses import dataclass

import numpy as np

_TAG_END = 0


@dataclass(frozen=True)
class Byte:
    v: int


@dataclass(frozen=True)
class Short:
    v: int


@dataclass(frozen=True)
class Long:
    v: int


@dataclass(frozen=True)
class Float:
    v: float


@dataclass(frozen=True)
class Double:
    v: float


def _tag_id(value: object) -> int:
    if isinstance(value, Byte):
        return 1
    if isinstance(value, Short):
        return 2
    if isinstance(value, int):
        return 3
    if isinstance(value, Long):
        return 4
    if isinstance(value, Float):
        return 5
    if isinstance(value, (Double, float)):
        return 6
    if isinstance(value, np.ndarray):
        if value.dtype == np.int8:
            return 7
        if value.dtype == np.int32:
            return 11
        if value.dtype == np.int64:
            return 12
        raise ValueError(f"Unsupported array dtype: {value.dtype}")
    if isinstance(value, str):
        return 8
    if isinstance(value, list):
        return 9
    if isinstance(value, dict):
        return 10
    raise ValueError(f"Cannot map value to an NBT tag: {type(value)!r}")


def _java_utf8(text: str) -> bytes:
    """Encode like Java's writeUTF: NUL as C0 80, astral chars as CESU-8 pairs."""
    out = bytearray()
    for ch in text:
        code = ord(ch)
        if code < 0x80:
            if code == 0:
                out += b"\xc0\x80"
            else:
                out.append(code)
        elif code < 0x800:
            out.append(0xC0 | code >> 6)
            out.append(0x80 | code & 0x3F)
        elif code < 0x10000:
            out.append(0xE0 | code >> 12)
            out.append(0x80 | (code >> 6) & 0x3F)
            out.append(0x80 | code & 0x3F)
        else:
            code -= 0x10000
            for unit in (0xD800 | code >> 10, 0xDC00 | code & 0x3FF):
                out.append(0xE0 | unit >> 12)
                out.append(0x80 | (unit >> 6) & 0x3F)
                out.append(0x80 | unit & 0x3F)
    return bytes(out)


def _payload(value: object) -> bytes:
    if isinstance(value, Byte):
        return struct.pack(">b", value.v)
    if isinstance(value, Short):
        return struct.pack(">h", value.v)
    if isinstance(value, int):
        return struct.pack(">i", value)
    if isinstance(value, Long):
        return struct.pack(">q", value.v)
    if isinstance(value, Float):
        return struct.pack(">f", value.v)
    if isinstance(value, Double):
        return struct.pack(">d", value.v)
    if isinstance(value, float):
        return struct.pack(">d", value)
    if isinstance(value, np.ndarray):
        big_endian = value.dtype.newbyteorder(">")
        return struct.pack(">i", len(value)) + value.astype(big_endian).tobytes()
    if isinstance(value, str):
        raw = _java_utf8(value)
        return struct.pack(">H", len(raw)) + raw
    if isinstance(value, list):
        element_tag = _tag_id(value[0]) if value else _TAG_END
        head = struct.pack(">Bi", element_tag, len(value))
        return head + b"".join(_payload(v) for v in value)
    if isinstance(value, dict):
        out = bytearray()
        for key, item in value.items():
            raw = _java_utf8(key)
            out += struct.pack(">BH", _tag_id(item), len(raw))
            out += raw + _payload(item)
        out += b"\x00"
        return bytes(out)
    raise ValueError(f"Cannot serialize value: {type(value)!r}")


def write_nbt(root: dict, *, gzip: bool = True, root_name: str = "") -> bytes:
    raw = _java_utf8(root_name)
    data = struct.pack(">BH", 10, len(raw)) + raw + _payload(root)
    return gzip_module.compress(data) if gzip else data
