"""Resource pack loading with an mtime-keyed extraction cache."""

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.extract import (
    _TEX_SIZE,
    _WORLD_FACES,
    _CacheWriteError,
    _extract_pack,
    _normalize_state_key,
)

__all__ = ["Textures", "load_pack"]

_CACHE_SUFFIX = ".litematic-tex.npz"


class Textures(NamedTuple):
    """Pack textures resolved from the extraction cache."""

    version: str
    arrays: dict[str, np.ndarray]
    maps: dict[str, dict[str, str]]
    pack_mtime: float

    def faces(
        self, name: str, properties: Mapping[str, str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return up/east/south arrays for a block state, [] as fallback."""
        for state_key in (_state_key_of(properties), ""):
            entry = self.maps.get(f"{name}[{state_key}]")
            if entry is None or any(f not in entry for f in _WORLD_FACES):
                continue
            up = self.arrays.get(entry["up"])
            east = self.arrays.get(entry["east"])
            south = self.arrays.get(entry["south"])
            if up is not None and east is not None and south is not None:
                return (up, east, south)
        return None


def _state_key_of(properties: Mapping[str, str]) -> str:
    joined = ",".join(f"{key}={value}" for key, value in properties.items())
    return _normalize_state_key(joined)


def load_pack(pack_path: str | Path) -> Textures:
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
    cached = _read_cache(cache, mtime)
    if cached is not None:
        return cached
    try:
        _extract_pack(pack_file, cache, pack_mtime=mtime)
    except _CacheWriteError:
        cache = Path(tempfile.gettempdir()) / cache.name
        _extract_pack(pack_file, cache, pack_mtime=mtime)
    loaded = _parse_textures(cache)
    if loaded is None or loaded.pack_mtime != mtime:
        raise RenderError(f"load: cache unreadable after extraction: {cache}")
    return loaded


def _read_cache(cache: Path, mtime: float) -> Textures | None:
    if not cache.is_file():
        return None
    parsed = _parse_textures(cache)
    if parsed is None or parsed.version != "pack" or parsed.pack_mtime != mtime:
        return None
    return parsed


def _parse_textures(path: Path) -> Textures | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            version = str(data["meta/version"].item())
            pack_mtime = float(data["meta/pack_mtime"].item())
            arrays: dict[str, np.ndarray] = {}
            maps: dict[str, dict[str, str]] = {}
            for key in data.files:
                if key.startswith("tex/"):
                    array = data[key]
                    if array.shape == (_TEX_SIZE, _TEX_SIZE, 4):
                        if array.dtype == np.uint8:
                            arrays[key[len("tex/") :]] = array
                elif key.startswith("map/"):
                    block_state, _, face = key[len("map/") :].rpartition("/")
                    maps.setdefault(block_state, {})[face] = str(data[key].item())
        return Textures(
            version=version, arrays=arrays, maps=maps, pack_mtime=pack_mtime
        )
    except Exception:
        return None
