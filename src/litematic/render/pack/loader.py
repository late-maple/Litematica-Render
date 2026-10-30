"""Resource pack loading with an mtime-keyed extraction cache."""

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.pack.blockstate import (
    _normalize_state_key,
    _state_pairs,
    _when_matches,
)
from litematic.render.pack.extract import (
    _NPZ_VERSION,
    _CacheWriteError,
    _extract_pack,
)
from litematic.render.quads import QUAD_DTYPE, _redstone_tint

__all__ = ["Textures", "load_pack"]

_CACHE_SUFFIX = ".litematic-tex.npz"

# In-process cache keyed (path, mtime); skips re-reading the npz per render.
_PACK_CACHE: dict[tuple[str, float], "Textures"] = {}


class Textures(NamedTuple):
    """Pack textures and per-state quads resolved from the extraction cache."""

    version: str
    arrays: dict[str, np.ndarray]
    quads_map: dict[str, np.ndarray]
    occl_map: dict[str, bool]
    pack_mtime: float
    states: dict[str, dict[str, dict[str, str]]] | None
    parts_map: dict[str, str]
    mquad_map: dict[str, np.ndarray]

    def quads(
        self, name: str, properties: Mapping[str, str]
    ) -> np.ndarray | None:
        """Return the state's QUAD_DTYPE quads, [] key as fallback."""
        for state_key in self._matching_states(name, properties):
            entry = self.quads_map.get(f"{name}[{state_key}]")
            if entry is not None:
                return entry
        record = self.parts_map.get(name)
        if record is not None:
            merged = self._evaluated_parts(name, record, properties)
            if merged is not None:
                return merged
        return self.quads_map.get(f"{name}[]")

    def _evaluated_parts(
        self,
        name: str,
        record: str,
        properties: Mapping[str, str],
    ) -> np.ndarray | None:
        """Concatenate the parts whose condition the live state satisfies."""
        selected = [
            self.mquad_map[part["quads"]]
            for part in json.loads(record)["parts"]
            if part["when"] is None
            or _when_matches(part["when"], properties)
        ]
        if not selected:
            return None
        merged = (
            np.concatenate(selected) if len(selected) > 1 else selected[0]
        )
        if name.partition(":")[2] == "redstone_wire":
            merged = _power_tinted(merged, properties, self.arrays)
        return merged

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


def _power_tinted(
    quads: np.ndarray,
    properties: Mapping[str, str],
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    """Rewrite wire part texkeys to the live power's dust tint."""
    try:
        power = max(0, min(15, int(properties.get("power", "0"))))
    except (TypeError, ValueError):
        power = 0
    r, g, b = _redstone_tint(power)
    out = quads.copy()
    keys = []
    for entry in out["texkey"]:
        base = str(entry).partition("@")[0]
        candidate = f"{base}@{r},{g},{b}"
        keys.append(candidate if candidate in arrays else str(entry))
    out["texkey"] = np.asarray(keys, dtype=QUAD_DTYPE["texkey"])
    return out


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
        # A quads entry referencing an unknown tex array is unusable; drop
        # it so the render pipeline degrades to builtin shapes or magenta.
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
        return Textures(
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
