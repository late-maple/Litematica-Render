"""Per-state block resolution: pack quads, builtin shapes, texture backfill."""

import json
from collections.abc import Iterable, Mapping
from typing import NamedTuple

import numpy as np

from litematic.parse.model import BlockState
from litematic.render.errors import RenderError
from litematic.render.pack.blockstate import (
    _normalize_state_key,
    _state_pairs,
    _when_matches,
)
from litematic.render.quads import QUAD_DTYPE, _redstone_tint, unit_cube_quads
from litematic.render.shapes import builtin_quads

__all__ = ["MISS_NO_GEOMETRY", "MISS_NO_TEXTURE", "Resolution", "Resolver"]

MISS_NO_GEOMETRY = "no-geometry"
MISS_NO_TEXTURE = "no-texture"

_MAGENTA_TEXKEY = "__magenta__"
# Synthetic key the extractor writes the compiled vanilla frame pad under.
FRAME_PAD_KEY = "__frame_pad"
_MAGENTA_ARRAY = np.full((16, 16, 4), (255, 0, 255, 255), np.uint8)
_MAGENTA_ARRAY.flags.writeable = False  # painter must never mutate in place


class Resolution(NamedTuple):
    """One state's drawable outcome; quads are always paintable."""

    quads: np.ndarray  # (Q,) QUAD_DTYPE; magenta cube when the chain misses
    occludes: bool
    miss: str | None  # None, MISS_NO_GEOMETRY, or MISS_NO_TEXTURE


class Resolver:
    """Resolve palette states to paintable quads; the constructor is a test seam."""

    def __init__(
        self,
        version: str,
        arrays: dict[str, np.ndarray],
        quads_map: dict[str, np.ndarray],
        occl_map: dict[str, bool],
        pack_mtime: float | tuple[float, ...],
        states: dict[str, dict[str, dict[str, str]]] | None,
        parts_map: dict[str, str],
        mquad_map: dict[str, np.ndarray],
        item_display: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.version = version
        self.arrays = arrays
        self.quads_map = quads_map
        self.occl_map = occl_map
        self.pack_mtime = pack_mtime
        self.states = states
        self.parts_map = parts_map
        self.mquad_map = mquad_map
        self.item_display = item_display or {}

    def resolve(self, state: BlockState) -> Resolution:
        """Run the chain: pack quads, builtin shapes, backfill, magenta."""
        name = state.name
        properties = state.properties
        quads = self._pack_quads(name, properties)
        if quads is not None:
            return Resolution(quads, self._occludes(name, properties), None)
        miss = MISS_NO_GEOMETRY
        for props in (properties, {}):
            built = builtin_quads(name, props)
            if built is None:
                continue
            miss = MISS_NO_TEXTURE
            if _backfill_builtin(name, built, self.arrays):
                return Resolution(
                    built, self._occludes(name, properties), None
                )
            break
        return Resolution(unit_cube_quads(_MAGENTA_TEXKEY), False, miss)

    def texkey_arrays(self, keys: Iterable[str]) -> dict[str, np.ndarray]:
        """Array per texkey, magenta included; unknown keys are an error."""
        out: dict[str, np.ndarray] = {}
        for key in keys:
            if key == _MAGENTA_TEXKEY:
                out[key] = _MAGENTA_ARRAY
            elif key in self.arrays:
                out[key] = self.arrays[key]
            else:
                raise RenderError(
                    f"resolve: quad texkey missing from pack arrays: {key}"
                )
        return out

    def pack_quads(self, name: str) -> np.ndarray | None:
        """The stateless pack quads for a block, or None when absent."""
        return self._pack_quads(name, {})

    def _pack_quads(
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

    def _occludes(
        self, name: str, properties: Mapping[str, str]
    ) -> bool:
        """Return the state's opaque-cube flag; missing entries are False."""
        for state_key in self._matching_states(name, properties):
            flag = self.occl_map.get(f"{name}[{state_key}]")
            if flag is not None:
                return flag
        return False

    def _matching_states(
        self, name: str, properties: Mapping[str, str]
    ) -> list[str]:
        """State keys satisfied by the properties, most specific first."""
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


# Base-material conventions: longest suffix first, woods to planks, dyes to wool.
_FAMILY_SUFFIXES: tuple[str, ...] = (
    "_pressure_plate",
    "_fence_gate",
    "_wall_banner",
    "_wall_sign",
    "_trapdoor",
    "_button",
    "_stairs",
    "_fence",
    "_carpet",
    "_slab",
    "_wall",
    "_sign",
    "_banner",
    "_door",
)
_WOODS = frozenset(
    {
        "oak",
        "spruce",
        "birch",
        "jungle",
        "acacia",
        "dark_oak",
        "mangrove",
        "cherry",
        "bamboo",
        "crimson",
        "warped",
    }
)
_DYES = frozenset(
    {
        "white",
        "orange",
        "magenta",
        "light_blue",
        "yellow",
        "lime",
        "pink",
        "gray",
        "light_gray",
        "cyan",
        "purple",
        "blue",
        "brown",
        "green",
        "red",
        "black",
    }
)

_TINTS: dict[str, tuple[int, int, int]] = {
    "grass": (145, 189, 89),
    "foliage": (119, 171, 47),
    "water": (63, 118, 228),
}


def _tint_for(block_id: str, state_key: str) -> tuple[int, int, int] | None:
    path = block_id.split(":", 1)[-1]
    if path == "water" or path == "bubble_column":
        return _TINTS["water"]
    if path == "redstone_wire":
        # Empty state key = multipart part: live power rewrites texkeys later.
        if not state_key:
            return None
        return _redstone_state_tint(state_key)
    if "leaves" in path or path == "vine":
        return _TINTS["foliage"]
    if "grass" in path or "fern" in path:
        return _TINTS["grass"]
    return None


def _redstone_state_tint(state_key: str) -> tuple[int, int, int]:
    power = 0
    for pair in state_key.split(","):
        if pair.startswith("power="):
            try:
                power = max(0, min(15, int(pair[len("power=") :])))
            except ValueError:
                power = 0
    return _redstone_tint(power)


def _backfill_builtin(
    name: str, quads: np.ndarray, arrays: dict[str, np.ndarray]
) -> bool:
    """Fill builtin quads' texkeys from direct, then material, keys."""
    if np.all(quads["texkey"] != ""):
        return all(str(key) in arrays for key in quads["texkey"])
    ns, _, stem = name.partition(":")
    prefix = "" if ns in ("", "minecraft") else f"{ns}:"
    base = f"{prefix}block/{stem or name}"
    tint = _tint_for(name, "")
    direct = [base]
    if tint is not None:
        direct.insert(0, f"{base}@{tint[0]},{tint[1]},{tint[2]}")
    for key in direct:
        if key in arrays:
            quads["texkey"] = key
            return True
    door = _door_texkeys(prefix, stem)
    if door is not None and all(key in arrays for key in door):
        quads["texkey"] = np.where(
            quads["vtx"][..., 1].mean(axis=1) >= 8.0, door[0], door[1]
        )
        return True
    for key in _material_keys(prefix, stem):
        if key in arrays:
            quads["texkey"] = key
            return True
    return False


def _door_texkeys(prefix: str, stem: str) -> tuple[str, str] | None:
    """(top, bottom) texture keys for a vanilla door stem, else None."""
    if not stem.endswith("_door"):
        return None
    material = stem[: -len("_door")]
    return (
        f"{prefix}block/{material}_door_top",
        f"{prefix}block/{material}_door_bottom",
    )


def _material_keys(prefix: str, stem: str) -> tuple[str, ...]:
    """Ordered base-material texture keys per vanilla naming rules."""
    if stem.endswith("_pane"):
        # Vanilla panes texture from <stem>_top, then the bare material.
        material = stem[: -len("_pane")]
        return (
            f"{prefix}block/{stem}_top",
            f"{prefix}block/{material}",
            f"{prefix}block/{material}s",
        )
    for suffix in _FAMILY_SUFFIXES:
        if stem.endswith(suffix):
            material = stem[: -len(suffix)]
            break
    else:
        return ()
    if material in _WOODS:
        return (f"{prefix}block/{material}_planks",)
    if suffix == "_carpet" and material in _DYES:
        return (f"{prefix}block/{material}_wool",)
    return (f"{prefix}block/{material}", f"{prefix}block/{material}s")
