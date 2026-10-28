"""Resource pack loading with an mtime-keyed extraction cache."""

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.extract import (
    _NPZ_VERSION,
    _TEX_SIZE,
    _CacheWriteError,
    _extract_pack,
    _normalize_state_key,
)
from litematic.render.quads import QUAD_DTYPE

__all__ = ["Textures", "load_pack"]

_CACHE_SUFFIX = ".litematic-tex.npz"


class Textures(NamedTuple):
    """Pack textures and per-state quads resolved from the extraction cache."""

    version: str
    arrays: dict[str, np.ndarray]
    quads_map: dict[str, np.ndarray]
    occl_map: dict[str, bool]
    pack_mtime: float
    states: dict[str, dict[str, dict[str, str]]] | None = None

    def quads(
        self, name: str, properties: Mapping[str, str]
    ) -> np.ndarray | None:
        """Return the state's QUAD_DTYPE quads, [] key as fallback."""
        for state_key in self._matching_states(name, properties):
            entry = self.quads_map.get(f"{name}[{state_key}]")
            if entry is not None:
                return entry
        return None

    def occludes(self, name: str, properties: Mapping[str, str]) -> bool:
        """Return the state's opaque-cube flag; missing entries are False."""
        for state_key in self._matching_states(name, properties):
            flag = self.occl_map.get(f"{name}[{state_key}]")
            if flag is not None:
                return flag
        return False

    def _matching_states(
        self, name: str, properties: Mapping[str, str]
    ) -> list[str]:
        """State keys satisfied by the properties, most specific first.

        npz keys are the pack's own variant keys, whose property pairs act
        as requirements (vanilla partial-key semantics): a key matches when
        every one of its pairs equals the state's property value, so extra
        schematic properties such as waterlogged=false never break a match.
        """
        if self.states is None:
            return [_state_key_of(properties), ""]
        found = [
            (-len(pairs), key)
            for key, pairs in self.states.get(name, {}).items()
            if all(
                properties.get(prop) == value for prop, value in pairs.items()
            )
        ]
        return [key for _, key in sorted(found)]


def _state_key_of(properties: Mapping[str, str]) -> str:
    joined = ",".join(f"{key}={value}" for key, value in properties.items())
    return _normalize_state_key(joined)


def _state_index(
    quads_map: dict[str, np.ndarray], occl_map: dict[str, bool]
) -> dict[str, dict[str, dict[str, str]]]:
    """Map block name to {state key: required property pairs}."""
    index: dict[str, dict[str, dict[str, str]]] = {}
    for full_key in (*quads_map, *occl_map):
        block, _, state = str(full_key).partition("[")
        if not state.endswith("]"):
            continue
        index.setdefault(block, {})[state[:-1]] = _state_pairs(state[:-1])
    return index


def _state_pairs(state_key: str) -> dict[str, str]:
    """Parse a state key into its property pairs; the empty key is {}."""
    pairs: dict[str, str] = {}
    for part in state_key.split(","):
        prop, sep, value = part.partition("=")
        if sep:
            pairs[prop] = value
    return pairs


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
    if (
        parsed is None
        or parsed.version != _NPZ_VERSION
        or parsed.pack_mtime != mtime
    ):
        return None
    return parsed


def _parse_textures(path: Path) -> Textures | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            version = str(data["meta/version"].item())
            pack_mtime = float(data["meta/pack_mtime"].item())
            arrays: dict[str, np.ndarray] = {}
            quads_map: dict[str, np.ndarray] = {}
            occl_map: dict[str, bool] = {}
            for key in data.files:
                if key.startswith("tex/"):
                    array = data[key]
                    if array.shape == (_TEX_SIZE, _TEX_SIZE, 4):
                        if array.dtype == np.uint8:
                            arrays[key[len("tex/") :]] = array
                elif key.startswith("quads/"):
                    array = data[key]
                    if array.ndim == 1 and array.dtype == QUAD_DTYPE:
                        quads_map[key[len("quads/") :]] = array
                elif key.startswith("occl/"):
                    array = data[key]
                    if array.shape == () and array.dtype == np.uint8:
                        occl_map[key[len("occl/") :]] = bool(array.item())
        # A quads entry referencing an unknown tex array is unusable; drop
        # it so the render pipeline degrades to builtin shapes or magenta.
        quads_map = {
            key: value
            for key, value in quads_map.items()
            if {str(quad["texkey"]) for quad in value} <= arrays.keys()
        }
        return Textures(
            version=version,
            arrays=arrays,
            quads_map=quads_map,
            occl_map=occl_map,
            pack_mtime=pack_mtime,
            states=_state_index(quads_map, occl_map),
        )
    except Exception:
        return None
