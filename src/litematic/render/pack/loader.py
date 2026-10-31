"""Pack stack loading with an mtime- and code-keyed extraction cache."""

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.pack.extract import (
    _NPZ_VERSION,
    _CacheWriteError,
    _code_fingerprint,
    _extract_pack,
)
from litematic.render.pack.resolve import Resolver, _state_index
from litematic.render.quads import QUAD_DTYPE

__all__ = ["load_pack"]

_CACHE_SUFFIX = ".litematic-tex.npz"
_BAD_CHARS = r'[<>:"/\\|?*]'
_CACHE_STEM_MAX = 80  # the digest carries uniqueness, stems stay short

# In-process cache keyed (paths, mtimes); skips re-reading the npz per render.
_PACK_CACHE: dict[tuple[tuple[str, ...], tuple[float, ...]], Resolver] = {}


def load_pack(
    pack_path: str | Path, extra_packs: Sequence[str | Path] = ()
) -> Resolver:
    """Load the pack stack, extracting to an mtime- and code-keyed cache on miss."""
    packs = [Path(pack_path), *_pack_paths(extra_packs)]
    mtimes: list[float] = []
    resolved: list[str] = []
    for path in packs:
        if not path.is_file():
            raise RenderError(f"load: not a resource pack: {path}: no such file")
        try:
            mtimes.append(path.stat().st_mtime)
            resolved.append(str(path.resolve()))
        except OSError as exc:
            raise RenderError(
                f"load: not a resource pack: {path}: {exc}"
            ) from exc
    stacked = tuple(mtimes)
    key = (tuple(resolved), stacked)
    hit = _PACK_CACHE.get(key)
    if hit is not None:
        return hit
    if len(packs) == 1:
        cache = packs[0].with_name(packs[0].stem + _CACHE_SUFFIX)
    else:
        cache = packs[0].with_name(_stack_cache_name(packs, resolved))
    cached = _read_cache(cache, stacked)
    if cached is not None:
        _PACK_CACHE[key] = cached
        return cached
    try:
        _extract_pack(packs, cache, pack_mtime=stacked)
    except _CacheWriteError:
        cache = Path(tempfile.gettempdir()) / cache.name
        _extract_pack(packs, cache, pack_mtime=stacked)
    loaded = _parse_textures(cache)
    if (
        loaded is None
        or loaded.pack_mtime != stacked
        or loaded.code_hash != _code_fingerprint()
    ):
        raise RenderError(f"load: cache unreadable after extraction: {cache}")
    _PACK_CACHE[key] = loaded
    return loaded


def _pack_paths(extra_packs: Sequence[str | Path]) -> list[Path]:
    """Normalize the add-on list; a bare path becomes a one-entry stack."""
    if isinstance(extra_packs, (str, Path)):
        return [Path(extra_packs)]
    try:
        items = list(extra_packs)
    except TypeError as exc:
        raise RenderError(
            "load: extra_packs must be a sequence of paths"
        ) from exc
    for item in items:
        if not isinstance(item, (str, Path)):
            raise RenderError("load: extra_packs must be a sequence of paths")
    return [Path(item) for item in items]


def _stack_cache_name(packs: list[Path], resolved: list[str]) -> str:
    """Cache file name spanning the whole stack, disambiguated by a path hash."""
    digest = hashlib.sha1("\n".join(resolved).encode("utf-8")).hexdigest()[:8]
    stem = re.sub(_BAD_CHARS, "_", "+".join(p.stem for p in packs))
    return f"{stem[:_CACHE_STEM_MAX]}.{digest}{_CACHE_SUFFIX}"


def _read_cache(cache: Path, mtimes: tuple[float, ...]) -> Resolver | None:
    if not cache.is_file():
        return None
    parsed = _parse_textures(cache)
    if (
        parsed is None
        or parsed.version != _NPZ_VERSION
        or parsed.pack_mtime != mtimes
        or parsed.code_hash != _code_fingerprint()
    ):
        return None
    return parsed


def _parse_textures(path: Path) -> Resolver | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            version = str(data["meta/version"].item())
            pack_mtime = tuple(
                float(value) for value in np.atleast_1d(data["meta/pack_mtime"])
            )
            arrays: dict[str, np.ndarray] = {}
            quads_map: dict[str, np.ndarray] = {}
            occl_map: dict[str, bool] = {}
            parts_map: dict[str, str] = {}
            mquad_map: dict[str, np.ndarray] = {}
            item_display: dict[str, dict[str, str]] = {}
            code_hash = (
                str(data["meta/code_hash"].item())
                if "meta/code_hash" in data.files
                else ""
            )
            if "items" in data.files:
                try:
                    parsed = json.loads(str(data["items"].item()))
                    if isinstance(parsed, dict):
                        item_display = parsed
                except Exception:
                    item_display = {}
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
            item_display=item_display,
            code_hash=code_hash,
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
