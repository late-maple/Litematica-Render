"""Batch multi-angle API: bitwise identity to serial, order, dedup, errors."""

import json
import zipfile

import numpy as np
import pytest
from helpers.litematica_packer import build_litematic_bytes
from helpers.pack_fixtures import (
    _TWO,
    _png_bytes,
    _schem,
    _two_color_pack,
)

from litematic import loads
from litematic.render import (
    RenderError,
    isometric,
    isometric_batch,
    isometric_batch_png,
    to_png,
)


def test_single_view_batch_is_bitwise_serial(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    serial = isometric(
        schem, textures_path=pack, rotation=160.0, slant=25.0, max_size=256
    )
    batch = isometric_batch(
        schem, views=[(160.0, 25.0)], textures_path=pack, max_size=256
    )
    assert len(batch) == 1
    assert batch[0].shape == serial.shape
    assert batch[0].dtype == serial.dtype
    assert batch[0].tobytes() == serial.tobytes()


def test_duplicate_views_share_one_reference(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    batch = isometric_batch(
        schem, views=[(160.0, 25.0), (45.0, 30.0), (160.0, 25.0)],
        textures_path=pack, max_size=256,
    )
    assert len(batch) == 3
    assert batch[0] is batch[2]
    assert batch[0] is not batch[1]
    assert batch[1].tobytes() == isometric(
        schem, textures_path=pack, rotation=45.0, slant=30.0, max_size=256
    ).tobytes()


def test_all_views_validated_before_pack_work(tmp_path):
    schem = _schem(_TWO)
    with pytest.raises(RenderError, match="slant must be in"):
        isometric_batch(
            schem,
            views=[(160.0, 25.0), (160.0, 90.5)],
            textures_path=tmp_path / "missing.zip",
        )


def test_empty_views_rejected(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    with pytest.raises(RenderError, match="views"):
        isometric_batch(_schem(_TWO), views=[], textures_path=pack)


def test_malformed_views_rejected(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    for bad in (None, [(160.0,)], [(160.0, 25.0, 5.0)], [42]):
        with pytest.raises(RenderError, match="views"):
            isometric_batch(schem, views=bad, textures_path=pack)


def test_int_and_float_same_angle_dedup_together(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    batch = isometric_batch(
        schem, views=[(160, 25.0), (160.0, 25)], textures_path=pack, max_size=256
    )
    assert batch[0] is batch[1]


def test_batch_png_matches_serial_bytes(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    serial = isometric(
        schem, textures_path=pack, rotation=160.0, slant=25.0, max_size=256
    )
    batch = isometric_batch_png(
        schem, views=[(160.0, 25.0), (45.0, 30.0), (160.0, 25.0)],
        textures_path=pack, max_size=256,
    )
    assert len(batch) == 3
    assert isinstance(batch[0], bytes)
    assert batch[0] == to_png(serial)
    assert batch[0] is batch[2]
    assert batch[0] != batch[1]


def test_batch_png_error_surface_matches_batch(tmp_path):
    with pytest.raises(RenderError, match="views"):
        isometric_batch_png(_schem(_TWO), views=[], textures_path=None)
    with pytest.raises(RenderError, match="slant must be in"):
        isometric_batch_png(
            _schem(_TWO), views=[(160.0, 90.5)], textures_path=None
        )


def test_empty_scene_raises_serial_error(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = loads(
        build_litematic_bytes(
            [
                {
                    "name": "Main",
                    "position": (0, 0, 0),
                    "size": (1, 1, 1),
                    "palette": [("minecraft:air", {})],
                    "indices_yzx": np.zeros(1, np.uint16),
                }
            ]
        )
    )
    with pytest.raises(RenderError, match="no blocks to render"):
        isometric_batch(
            schem, views=[(160.0, 25.0), (45.0, 30.0)], textures_path=pack
        )


def test_four_directions_match_serial_in_order(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    views = [(45.0, 35.264389682754654), (135.0, 25.0), (225.0, 25.0), (315.0, 25.0)]
    batch = isometric_batch(
        schem, views=views, textures_path=pack, max_size=256
    )
    assert len(batch) == 4
    for frame, (rotation, slant) in zip(batch, views):
        serial = isometric(
            schem, textures_path=pack, rotation=rotation, slant=slant, max_size=256
        )
        assert frame.shape == serial.shape
        assert frame.tobytes() == serial.tobytes()


def test_same_tier_set_views_are_bitwise_serial(tmp_path):
    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    iso = 35.264389682754654
    views = [(45.0, 25.0), (45.0, iso), (45.0, 35.0)]
    batch = isometric_batch(schem, views=views, textures_path=pack, max_size=256)
    assert len(batch) == 3
    for frame, (rotation, slant) in zip(batch, views):
        serial = isometric(
            schem, textures_path=pack, rotation=rotation, slant=slant, max_size=256
        )
        assert frame.shape == serial.shape
        assert frame.tobytes() == serial.tobytes()


def _south_only_pack(path):
    entries = {
        "assets/minecraft/blockstates/stone.json": json.dumps(
            {"variants": {"": {"model": "block/stone"}}}
        ),
        "assets/minecraft/models/block/stone.json": json.dumps(
            {
                "elements": [
                    {
                        "from": [0, 0, 0],
                        "to": [16, 16, 16],
                        "faces": {"south": {"texture": "block/stone"}},
                    }
                ]
            }
        ),
        "assets/minecraft/textures/block/stone.png": _png_bytes((255, 0, 0)),
    }
    with zipfile.ZipFile(path, "w") as zf:
        for key, data in entries.items():
            zf.writestr(key, data)
    return path


def test_hidden_tier_group_raises_empty_scene(tmp_path):
    pack = _south_only_pack(tmp_path / "pack.zip")
    schem = _schem([("stone", 0, 0, 0)])
    with pytest.raises(RenderError, match="no blocks to render"):
        isometric_batch(
            schem, views=[(45.0, 35.264389682754654), (225.0, 35.264389682754654)],
            textures_path=pack, max_size=256,
        )


def test_compose_prepare_failure_wrapped(tmp_path, monkeypatch):
    import litematic.render as render_mod

    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    monkeypatch.setattr(
        render_mod, "_prepare", _boom
    )
    with pytest.raises(RenderError, match="compose"):
        isometric_batch(
            schem, views=[(160.0, 25.0)], textures_path=pack, max_size=256
        )


def test_compose_fill_failure_wrapped(tmp_path, monkeypatch):
    import litematic.render as render_mod

    pack = _two_color_pack(tmp_path / "pack.zip")
    schem = _schem(_TWO)
    monkeypatch.setattr(render_mod, "_fill", _boom)
    with pytest.raises(RenderError, match="compose"):
        isometric_batch(
            schem, views=[(160.0, 25.0)], textures_path=pack, max_size=256
        )


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def test_param_error_precedes_angle_error_like_serial():
    schem = _schem(_TWO)
    with pytest.raises(RenderError, match="max_size"):
        isometric(schem, textures_path=None, rotation="45", max_size=9000)
    with pytest.raises(RenderError, match="max_size"):
        isometric_batch(
            schem, views=[("45", 25.0)], textures_path=None, max_size=9000
        )
