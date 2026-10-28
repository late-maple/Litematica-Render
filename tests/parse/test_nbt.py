"""read_nbt contract: mapping, byte order, compression, corrupt input."""

import struct

import numpy as np
import pytest
from helpers.nbt_writer import Byte, Double, Float, Long, Short, write_nbt

from litematic import NbtError
from litematic.parse.nbt import read_nbt

TREE = {
    "DataVersion": 3337,
    "Name": "test schematic",
    "Byte": Byte(-7),
    "Short": Short(-300),
    "Long": Long(2**40),
    "Float": Float(1.5),
    "Double": Double(-2.25),
    "Nested": {"Inner": "value", "Deep": {"k": 0}},
    "IntList": [1, 2, 3],
    "StrList": ["a", "b"],
    "EmptyList": [],
    "Bytes": np.array([-1, 0, 127], dtype=np.int8),
    "Ints": np.array([1, -2, 2**31 - 1], dtype=np.int32),
    "Longs": np.array([1, 127, 2**40, -5], dtype=np.int64),
}


def _assert_equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_equal(a, e)
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, (Byte, Short, Long)):
        assert actual == expected.v
    elif isinstance(expected, (Float, Double)):
        assert actual == pytest.approx(expected.v)
    else:
        assert actual == expected


def test_roundtrip_gzip():
    _assert_equal(read_nbt(write_nbt(TREE)), TREE)


def test_roundtrip_raw():
    _assert_equal(read_nbt(write_nbt(TREE, gzip=False)), TREE)


def test_long_array_big_endian_values():
    arr = np.array([1, 127, 300, 2**40, -5], dtype=np.int64)
    out = read_nbt(write_nbt({"L": arr}))["L"]
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, arr)


def test_garbage_input_raises():
    with pytest.raises(NbtError):
        read_nbt(b"notnbt!")


def test_truncated_gzip_raises():
    data = write_nbt(TREE)
    with pytest.raises(NbtError):
        read_nbt(data[: len(data) // 2])


def test_truncated_raw_raises():
    data = write_nbt(TREE, gzip=False)
    with pytest.raises(NbtError):
        read_nbt(data[: len(data) // 2])


def test_corrupt_deflate_raises():
    # valid gzip header, then a stored block whose LEN/NLEN check fails
    data = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff" + b"\x00" * 8
    with pytest.raises(NbtError):
        read_nbt(data)


def _string_doc(raw: bytes) -> bytes:
    # root compound {S: <string with the given raw payload bytes>}
    return (
        b"\x0a\x00\x00"
        + b"\x08\x00\x01S"
        + struct.pack(">H", len(raw))
        + raw
        + b"\x00"
    )


def test_supplementary_char_from_cesu8_surrogate_pair():
    # U+1F3E0 written by Java writeUTF as surrogate pair D83C DFE0
    raw = b"\xed\xa0\xbc\xed\xbf\xa0"
    assert read_nbt(_string_doc(raw))["S"] == "\U0001F3E0"


def test_null_from_overlong_c080():
    # Java writeUTF encodes U+0000 as the overlong sequence C0 80
    raw = b"\xc0\x80"
    assert read_nbt(_string_doc(raw))["S"] == "\x00"


def test_java_modified_utf8_roundtrip():
    text = "emoji \U0001F3E0 nul \x00 end"
    assert read_nbt(write_nbt({"Name": text}))["Name"] == text


@pytest.mark.parametrize(
    "raw",
    [
        b"\xc0\x81",  # overlong encoding (not the legal C0 80 NUL)
        b"\x80",  # isolated continuation byte
        b"\xe0\x80",  # truncated 3-byte sequence
        b"\xed\xa0\xbd",  # lone high surrogate D83D
        b"\xed\xb8\xa0",  # lone low surrogate DC00
    ],
)
def test_malformed_java_utf8_raises_nbt_error(raw):
    with pytest.raises(NbtError):
        read_nbt(_string_doc(raw))


def test_malformed_key_name_raises_nbt_error():
    # entry key name contains an isolated continuation byte
    doc = (
        b"\x0a\x00\x00"
        + b"\x08\x00\x02"
        + b"S\x80"
        + b"\x00\x01"
        + b"x"
        + b"\x00"
    )
    with pytest.raises(NbtError):
        read_nbt(doc)


def _nested_compound_bytes(levels: int) -> bytes:
    # built inside-out so the test itself does not recurse; every compound
    # body is entry... + trailing end byte
    body = b"\x03\x00\x01N" + struct.pack(">i", 7) + b"\x00"
    for _ in range(levels):
        body = b"\x0a\x00\x01N" + body + b"\x00"
    return b"\x0a\x00\x00" + body


def test_deep_nesting_raises_nbt_error():
    with pytest.raises(NbtError):
        read_nbt(_nested_compound_bytes(5000))


def test_nesting_within_limit_parses():
    levels = 200
    node = read_nbt(_nested_compound_bytes(levels))
    for _ in range(levels):
        node = node["N"]
    assert node["N"] == 7
