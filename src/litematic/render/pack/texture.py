"""Texture resolution: refs, models, PNG decode, tints."""

import io
import json
import zipfile
from typing import Any

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.quads import redstone_tint

_VEC = {
    "up": (0, 1, 0),
    "down": (0, -1, 0),
    "north": (0, 0, -1),
    "south": (0, 0, 1),
    "west": (-1, 0, 0),
    "east": (1, 0, 0),
}
_NAME = {v: k for k, v in _VEC.items()}



class _Budget:
    """Reject oversize or over-budget pack textures as missing."""

    MAX_SIDE = 2048
    MAX_BYTES = 256 * 1024 * 1024

    def __init__(self) -> None:
        self.bytes = 0

    def take(self, shape: tuple[int, ...]) -> bool:
        """Account one decoded texture's bytes; False when over a cap."""
        height, width = int(shape[0]), int(shape[1])
        if max(height, width) > self.MAX_SIDE:
            return False
        size = height * width * 4
        if self.bytes + size > self.MAX_BYTES:
            return False
        self.bytes += size
        return True



_TINTS: dict[str, tuple[int, int, int]] = {
    "grass": (145, 189, 89),
    "foliage": (119, 171, 47),
    "water": (63, 118, 228),
}



def _resolve_ref(ref: Any, textures: dict[str, str]) -> str | None:
    """Resolve a face texture reference, following # chains up to depth 32."""
    if not isinstance(ref, str) or not ref:
        return None
    current = ref
    for _ in range(32):
        if not current.startswith("#"):
            return current
        nxt = textures.get(current[1:])
        if not isinstance(nxt, str):
            return None
        current = nxt
    return None



def _texture_key(ref: str, ns: str) -> str:
    ref_ns, _, path = ref.partition(":")
    if not path:
        ref_ns, path = ns, ref_ns
    if ref_ns == "minecraft":
        return path
    return f"{ref_ns}:{path}"



def _asset_path(ns: str, ref: str, kind: str, suffix: str) -> str | None:
    if ".." in ref:
        return None
    ref_ns, _, path = ref.partition(":")
    if not path:
        return f"assets/{ns}/{kind}/{ref_ns}{suffix}"
    path = path.removeprefix(f"{kind}/")
    return f"assets/{ref_ns}/{kind}/{path}{suffix}"



def _cached_model(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    ref: str,
    cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    key = f"{ns}|{ref}"
    if key not in cache:
        model = _resolve_model(archive, names, ns, ref)
        if model is None:
            cache[key] = None
        elif model["elements"] is None:
            stub_ref = _stub_texture(names, ns, model["textures"])
            cache[key] = (
                None
                if stub_ref is None
                else {
                    "textures": model["textures"],
                    "elements": None,
                    "stub_ref": stub_ref,
                    "builtin_cut": bool(model.get("builtin_cut")),
                }
            )
        else:
            cache[key] = model
    return cache[key]



def _stub_texture(
    names: set[str], ns: str, textures: dict[str, Any]
) -> str | None:
    """Find the first concrete texture ref an element-less model can load."""
    for value in textures.values():
        if not isinstance(value, str) or value.startswith("#"):
            continue
        path = _asset_path(ns, value, "textures", ".png")
        if path is not None and path in names:
            return value
    return None



def _resolve_model(
    archive: zipfile.ZipFile, names: set[str], ns: str, model_ref: str
) -> dict[str, Any] | None:
    """Walk the parent chain and merge textures; nearest elements win."""
    merged_textures: dict[str, str] = {}
    elements: list[dict[str, Any]] | None = None
    seen: set[str] = set()
    ref: str | None = model_ref
    while ref is not None and not ref.split(":")[-1].startswith("builtin"):
        if ref in seen:
            return None
        seen.add(ref)
        file = _asset_path(ns, ref, "models", ".json")
        if file is None or file not in names:
            return None
        model = json.loads(archive.read(file))
        if not isinstance(model, dict):
            return None
        textures = model.get("textures")
        if isinstance(textures, dict):
            for key, value in textures.items():
                merged_textures.setdefault(key, value)
        if elements is None and isinstance(model.get("elements"), list):
            elements = model["elements"]
        parent = model.get("parent")
        ref = parent if isinstance(parent, str) else None
    if elements is None:
        if ref is not None:
            # Chain terminated at a builtin/* parent: no pack geometry, but
            # the merged textures still feed the spec 5.3 texture fallback.
            return {
                "textures": merged_textures,
                "elements": None,
                "builtin_cut": True,
            }
        return {"textures": merged_textures, "elements": None}
    return {"textures": merged_textures, "elements": elements}



def _rotate(face: str, x: int, y: int) -> str:
    vec = _VEC[face]
    for _ in range((x // 90) % 4):  # y' = z, z' = -y
        vec = (vec[0], vec[2], -vec[1])
    for _ in range((y // 90) % 4):  # x' = -z, z' = x
        vec = (-vec[2], vec[1], vec[0])
    return _NAME[vec]



def _load_texture(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    tex_ref: str,
    tex_key: str,
    cache: dict[str, np.ndarray | None],
    budget: _Budget,
) -> np.ndarray | None:
    if tex_key in cache:
        return cache[tex_key]
    file = _asset_path(ns, tex_ref, "textures", ".png")
    result: np.ndarray | None = None
    if file is not None and file in names:
        try:
            image = _decode_png(archive.read(file), _is_animated(archive, names, file))
            raw = np.asarray(image, dtype=np.uint8)
            if budget.take(raw.shape):
                result = raw
        except RenderError:
            raise
        except Exception:
            result = None
    cache[tex_key] = result
    return result



def _is_animated(
    archive: zipfile.ZipFile, names: set[str], file: str
) -> bool:
    """True when the texture's .mcmeta declares a vanilla animation."""
    meta = f"{file}.mcmeta"
    if meta not in names:
        return False
    try:
        parsed = json.loads(archive.read(meta))
    except Exception:
        return False
    return isinstance(parsed, dict) and "animation" in parsed



def _decode_png(data: bytes, animated: bool) -> Any:
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGBA")
    if animated:  # vertical strip: first frame only
        width, height = image.size
        side = min(width, height)
        image = image.crop((0, 0, side, side))
    return image



def _apply_tint(array: np.ndarray, tint: tuple[int, int, int]) -> np.ndarray:
    factor = np.array(tint + (255,), dtype=np.uint16)
    return ((array.astype(np.uint16) * factor) // 255).astype(np.uint8)



def _tint_for(block_id: str, state_key: str) -> tuple[int, int, int] | None:
    path = block_id.split(":", 1)[-1]
    if path == "water" or path == "bubble_column":
        return _TINTS["water"]
    if path == "redstone_wire":
        # An empty state key marks the multipart part path: the dust tint
        # depends on the live power, so the base texture ships untinted
        # and pack.py rewrites the texkeys at lookup time.
        if not state_key:
            return None
        return _redstone_tint(state_key)
    if "leaves" in path or path == "vine":
        return _TINTS["foliage"]
    if "grass" in path or "fern" in path:
        return _TINTS["grass"]
    return None



def _redstone_tint(state_key: str) -> tuple[int, int, int]:
    power = 0
    for pair in state_key.split(","):
        if pair.startswith("power="):
            try:
                power = max(0, min(15, int(pair[len("power=") :])))
            except ValueError:
                power = 0
    return redstone_tint(power)



def _first_model_texture(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    model_refs: list[str],
    model_cache: dict[str, dict[str, Any] | None],
    texture_cache: dict[str, np.ndarray | None],
    budget: _Budget,
) -> np.ndarray | None:
    """First loadable concrete texture among the referenced models."""
    for ref in model_refs:
        model = _cached_model(archive, names, ns, ref, model_cache)
        if model is None:
            continue
        for value in model["textures"].values():
            tex_ref = _resolve_ref(value, model["textures"])
            if tex_ref is None:
                continue
            raw = _load_texture(
                archive,
                names,
                ns,
                tex_ref,
                _texture_key(tex_ref, ns),
                texture_cache,
                budget,
            )
            if raw is not None:
                return raw
    return None



def _register_builtin_texture(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    block_id: str,
    model_refs: list[str],
    model_cache: dict[str, dict[str, Any] | None],
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    budget: _Budget,
) -> None:
    """Register the builtin layer texture under the block's canonical key.

    Spec 5.3 chain: the block's own model textures first, then
    textures/block/<stem>.png; unresolved blocks stay for magenta.
    """
    raw = _first_model_texture(
        archive, names, ns, model_refs, model_cache, texture_cache, budget
    )
    stem = block_id.partition(":")[2]
    tex_key = _texture_key(f"block/{stem}", ns)
    if raw is None:
        raw = _load_texture(
            archive, names, ns, f"block/{stem}", tex_key, texture_cache, budget
        )
    if raw is None:
        return
    tex_arrays.setdefault(tex_key, raw)
    tint = _tint_for(block_id, "")
    if tint is not None:
        array_key = f"{tex_key}@{tint[0]},{tint[1]},{tint[2]}"
        if array_key not in tex_arrays:
            tex_arrays[array_key] = _apply_tint(raw, tint)



_WIRE_DUST_TEXTURES = ("block/redstone_dust_dot", "block/redstone_dust_line0")



def _register_wire_dust(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    budget: _Budget,
) -> None:
    """Register power-tinted dust textures for the builtin wire family."""
    for ref in _WIRE_DUST_TEXTURES:
        tex_key = _texture_key(ref, ns)
        raw = _load_texture(
            archive, names, ns, ref, tex_key, texture_cache, budget
        )
        if raw is None:
            continue
        for power in range(16):
            tint = redstone_tint(power)
            tex_arrays.setdefault(
                f"{tex_key}@{tint[0]},{tint[1]},{tint[2]}",
                _apply_tint(raw, tint),
            )
