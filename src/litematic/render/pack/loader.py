"""Resource pack loading with an mtime-keyed extraction cache."""

import json
import tempfile
from pathlib import Path

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.pack.extract import (
    _NPZ_VERSION,
    _CacheWriteError,
    _extract_pack,
)
from litematic.render.pack.resolve import Resolver, _state_index
from litematic.render.quads import QUAD_DTYPE

__all__ = ["load_pack"]

_CACHE_SUFFIX = ".litematic-tex.npz"

# In-process cache keyed (path, mtime); skips re-reading the npz per render.
_PACK_CACHE: dict[tuple[str, float], Resolver] = {}


def load_pack(pack_path: str | Path) -> Resolver:
    """Load pack textures, extracting to a mtime-keyed cache on miss."""
    pack_file = Path(pack_path)
    if not pack_file.is_file():
        raise RenderError(f"load: not a resource pack: {pack_file}: no such file")
    try:
        mtime = pack_file.stat().st_mtime
    except OSError as exc:
        raise RenderError(
            f"load: not a resource pack: {pack_file}: {exc}"
        ) from exc
    cache = pack_file.with_name(pack_file.stem + _CACHE_SUFFIX)
    key = (str(pack_file.resolve()), mtime)
    hit = _PACK_CACHE.get(key)
    if hit is not None:
        return hit
    cached = _read_cache(cache, mtime)
    if cached is not None:
        _PACK_CACHE[key] = cached
        return cached
    try:
        _extract_pack(pack_file, cache, pack_mtime=mtime)
    except _CacheWriteError:
        cache = Path(tempfile.gettempdir()) / cache.name
        _extract_pack(pack_file, cache, pack_mtime=mtime)
    loaded = _parse_textures(cache)
    if loaded is None or loaded.pack_mtime != mtime:
        raise RenderError(f"load: cache unreadable after extraction: {cache}")
    _PACK_CACHE[key] = loaded
    return loaded


def _read_cache(cache: Path, mtime: float) -> Resolver | None:
    if not cache.is_file():
        return None
    parsed = _parse_textures(cache)
    if (
        parsed is None
        or parsed.version != _NPZ_VERSION
        or parsed.pack_mtime != mtime
    ):
        return None
    return parsed


def _parse_textures(path: Path) -> Resolver | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            version = str(data["meta/version"].item())
            pack_mtime = float(data["meta/pack_mtime"].item())
            arrays: dict[str, np.ndarray] = {}
            quads_map: dict[str, np.ndarray] = {}
            occl_map: dict[str, bool] = {}
            parts_map: dict[str, str] = {}
            mquad_map: dict[str, np.ndarray] = {}
            for key in data.files:
                if key.startswith("tex/"):
                    array = data[key]
                    if (
                        array.ndim == 3
                        and array.shape[2] == 4
                        and min(array.shape[:2]) >= 1
                        and array.dtype == np.uint8
                    ):
                        arrays[key[len("tex/") :]] = array
                elif key.startswith("quads/"):
                    array = data[key]
                    if array.ndim == 1 and array.dtype == QUAD_DTYPE:
                        quads_map[key[len("quads/") :]] = array
                elif key.startswith("occl/"):
                    array = data[key]
                    if array.shape == () and array.dtype == np.uint8:
                        occl_map[key[len("occl/") :]] = bool(array.item())
                elif key.startswith("parts/"):
                    array = data[key]
                    if array.shape == ():
                        parts_map[key[len("parts/") :]] = str(array.item())
                elif key.startswith("mquads/"):
                    array = data[key]
                    if array.ndim == 1 and array.dtype == QUAD_DTYPE:
                        mquad_map[key[len("mquads/") :]] = array
        # Quads referencing unknown arrays are dropped; render degrades instead.
        quads_map = {
            key: value
            for key, value in quads_map.items()
            if {str(quad["texkey"]) for quad in value} <= arrays.keys()
        }
        mquad_map = {
            key: value
            for key, value in mquad_map.items()
            if {str(quad["texkey"]) for quad in value} <= arrays.keys()
        }
        parts_map = {
            key: record
            for key, record in parts_map.items()
            if _parts_usable(record, mquad_map)
        }
        return Resolver(
            version=version,
            arrays=arrays,
            quads_map=quads_map,
            occl_map=occl_map,
            pack_mtime=pack_mtime,
            states=_state_index(quads_map, occl_map),
            parts_map=parts_map,
            mquad_map=mquad_map,
        )
    except Exception:
        return None


def _parts_usable(record: str, mquad_map: dict[str, np.ndarray]) -> bool:
    """A parts record is usable when every referenced quad array exists."""
    try:
        parts = json.loads(record)["parts"]
        return all(
            isinstance(part.get("quads"), str) and part["quads"] in mquad_map
            for part in parts
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return False
