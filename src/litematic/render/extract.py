"""Resource pack extraction: blockstates → models → quads → npz."""

import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Callable, NamedTuple, cast

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.quads import (
    QUAD_DTYPE,
    TIER_DOWN,
    TIER_EAST,
    TIER_NORTH,
    TIER_SOUTH,
    TIER_UP,
    TIER_WEST,
    is_parallelogram,
    sort_quads,
    unit_cube_quads,
)

__all__ = ["ExtractSummary"]

_TEX_SIZE = 16
_FIXED_DATE = (1980, 1, 1, 0, 0, 0)
_NPZ_VERSION = "pack-quads"
_QUAD_FUSE = 256
_OPAQUE_ALPHA = 250.0
_TINTS: dict[str, tuple[int, int, int]] = {
    "grass": (145, 189, 89),
    "foliage": (119, 171, 47),
    "water": (63, 118, 228),
}
_REDSTONE_LOW = (76, 0, 0)
_REDSTONE_HIGH = (255, 26, 26)
_VEC = {
    "up": (0, 1, 0),
    "down": (0, -1, 0),
    "north": (0, 0, -1),
    "south": (0, 0, 1),
    "west": (-1, 0, 0),
    "east": (1, 0, 0),
}
_NAME = {v: k for k, v in _VEC.items()}
_FACE_TIER = {
    "down": TIER_DOWN,
    "up": TIER_UP,
    "north": TIER_NORTH,
    "south": TIER_SOUTH,
    "west": TIER_WEST,
    "east": TIER_EAST,
}
# Pinned auto-UV axes (v runs down the texture): down u=x v=16-z | up u=x
# v=z | north u=16-x v=16-y | south u=x v=16-y | west u=16-z v=16-y |
# east u=z v=16-y. Corner order is always (u1,v1),(u2,v1),(u2,v2),(u1,v2).
_FACE_GEOM: dict[int, tuple[tuple[int, int, int], ...]] = {
    TIER_DOWN: ((0, 0, 16), (16, 0, 16), (16, 0, 0), (0, 0, 0)),
    TIER_UP: ((0, 16, 0), (16, 16, 0), (16, 16, 16), (0, 16, 16)),
    TIER_NORTH: ((16, 16, 0), (0, 16, 0), (0, 0, 0), (16, 0, 0)),
    TIER_SOUTH: ((0, 16, 16), (16, 16, 16), (16, 0, 16), (0, 0, 16)),
    TIER_WEST: ((0, 16, 16), (0, 16, 0), (0, 0, 0), (0, 0, 16)),
    TIER_EAST: ((16, 16, 0), (16, 16, 16), (16, 0, 16), (16, 0, 0)),
}


class ExtractSummary(NamedTuple):
    """Outcome of one pack extraction."""

    path: Path
    blocks: int
    skipped: int


class _CacheWriteError(RenderError):
    """Raised when the npz destination cannot be written."""


def _extract_pack(
    source: str | Path, dest: str | Path, pack_mtime: float | None = None
) -> ExtractSummary:
    src = Path(source)
    if not src.is_file():
        raise RenderError(f"extract: not a resource pack: {src}: no such file")
    try:
        archive = zipfile.ZipFile(src)
    except zipfile.BadZipFile as exc:
        raise RenderError(f"extract: not a resource pack: {src}: {exc}") from exc
    if pack_mtime is None:
        try:
            pack_mtime = src.stat().st_mtime
        except OSError as exc:
            raise RenderError(f"extract: not a resource pack: {src}: {exc}") from exc
    with archive:
        names = set(archive.namelist())
        state_files = sorted(
            n
            for n in names
            if n.startswith("assets/") and "/blockstates/" in n and n.endswith(".json")
        )
        if not state_files and not _fallback_blocks(names):
            raise RenderError(
                "extract: not a resource pack: "
                f"{src}: no blockstates or block textures under assets/"
            )
        _require_pillow()
        texture_cache: dict[str, np.ndarray | None] = {}
        model_cache: dict[str, dict[str, Any] | None] = {}
        tex_arrays: dict[str, np.ndarray] = {}
        quad_map: dict[str, np.ndarray] = {}
        occl_map: dict[str, bool] = {}
        blocks = 0
        skipped = 0
        covered: set[str] = set()
        for state_name in state_files:
            try:
                states = _extract_block(
                    archive, names, state_name, model_cache, texture_cache, tex_arrays
                )
            except RenderError:
                raise
            except Exception:
                skipped += 1
                continue
            if not states:
                skipped += 1
                continue
            block_id = _block_id(state_name)
            covered.add(block_id)
            for state_key, (quads, occl) in states.items():
                key = f"{block_id}[{state_key}]"
                quad_map[key] = quads
                occl_map[key] = occl
            blocks += 1
        for ns, stem in _fallback_blocks(names):
            block_id = f"{ns}:{stem}"
            if block_id in covered:
                continue
            array_key = _fallback_array_key(
                archive, names, ns, stem, texture_cache, tex_arrays
            )
            if array_key is None:
                continue
            cube = unit_cube_quads(array_key)
            quad_map[f"{block_id}[]"] = cube
            occl_map[f"{block_id}[]"] = _unit_cube_occl(cube, tex_arrays)
            blocks += 1
    _write_npz(Path(dest), tex_arrays, quad_map, occl_map, pack_mtime)
    return ExtractSummary(Path(dest), blocks, skipped)


def _require_pillow() -> None:
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise RenderError(
            "extract: Pillow is required — pip install litematic-render[pack]"
        ) from exc


def _extract_block(
    archive: zipfile.ZipFile,
    names: set[str],
    state_name: str,
    model_cache: dict[str, dict[str, Any] | None],
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, bool]]:
    """Extract compiled quads and occl flags for every variant of a state."""
    state = json.loads(archive.read(state_name))
    variants = _all_variants(state)
    if not variants:
        return {}
    ns = state_name.split("/")[1]
    block_id = _block_id(state_name)
    result: dict[str, tuple[np.ndarray, bool]] = {}
    for state_key, variant in variants:
        x_rot = int(variant.get("x", 0))
        y_rot = int(variant.get("y", 0))
        model = _cached_model(archive, names, ns, variant["model"], model_cache)
        if model is None:
            continue
        quads = _variant_quads(
            archive,
            names,
            ns,
            model,
            block_id,
            state_key,
            x_rot,
            y_rot,
            texture_cache,
            tex_arrays,
        )
        if quads is None or len(quads) > _QUAD_FUSE:
            continue
        result[state_key] = (quads, _unit_cube_occl(quads, tex_arrays))
    return result


def _variant_quads(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    model: dict[str, Any],
    block_id: str,
    state_key: str,
    x_rot: int,
    y_rot: int,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Compile one variant's quads: element boxes, else a stub unit cube."""
    if model["elements"] is None:
        return _stub_quads(
            archive, names, ns, model, block_id, state_key, texture_cache, tex_arrays
        )

    def texkey_for(
        face: dict[str, Any], textures: dict[str, str], face_ns: str
    ) -> str | None:
        return _element_texkey(
            archive,
            names,
            face_ns,
            face,
            textures,
            block_id,
            state_key,
            texture_cache,
            tex_arrays,
        )

    return _compile_elements(
        model["elements"], model["textures"], ns, x_rot, y_rot, texkey_for
    )


def _element_texkey(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    face: dict[str, Any],
    textures: dict[str, str],
    block_id: str,
    state_key: str,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
) -> str | None:
    """Resolve one element face's texture ref into a tex array key."""
    tex_ref = _resolve_ref(face.get("texture"), textures)
    if tex_ref is None:
        return None
    return _ref_array_key(
        archive,
        names,
        ns,
        tex_ref,
        "tintindex" in face,
        block_id,
        state_key,
        texture_cache,
        tex_arrays,
    )


def _ref_array_key(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    tex_ref: str,
    tinted: bool,
    block_id: str,
    state_key: str,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
) -> str | None:
    """Load a texture ref and register its raw or tinted array key."""
    tex_key = _texture_key(tex_ref, ns)
    raw = _load_texture(archive, names, ns, tex_ref, tex_key, texture_cache)
    if raw is None:
        return None
    tint = _tint_for(block_id, state_key) if tinted else None
    return _register_array(raw, tex_key, tint, tex_arrays)


def _register_array(
    raw: np.ndarray,
    tex_key: str,
    tint: tuple[int, int, int] | None,
    tex_arrays: dict[str, np.ndarray],
) -> str:
    """Store a raw texture under tex_key, adding a tinted variant if asked."""
    if tint is None:
        tex_arrays.setdefault(tex_key, raw)
        return tex_key
    array_key = f"{tex_key}@{tint[0]},{tint[1]},{tint[2]}"
    if array_key not in tex_arrays:
        tex_arrays[array_key] = _apply_tint(raw, tint)
    return array_key


def _stub_quads(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    model: dict[str, Any],
    block_id: str,
    state_key: str,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Build unit-cube quads from an element-less model's first texture."""
    array_key = _ref_array_key(
        archive,
        names,
        ns,
        model["stub_ref"],
        True,
        block_id,
        state_key,
        texture_cache,
        tex_arrays,
    )
    if array_key is None:
        return None
    return unit_cube_quads(array_key)


def _fallback_array_key(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    stem: str,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
) -> str | None:
    """Resolve a textures/block/<stem>.png fallback into an array key."""
    tex_ref = f"block/{stem}"
    tex_key = _texture_key(tex_ref, ns)
    raw = _load_texture(archive, names, ns, tex_ref, tex_key, texture_cache)
    if raw is None:
        return None
    tint = _tint_for(f"{ns}:{stem}", "")
    return _register_array(raw, tex_key, tint, tex_arrays)


def _corner_tuple(point: Any) -> tuple[float, float, float]:
    """Convert one vertex row to a plain three-float tuple."""
    x, y, z = (float(value) for value in point)
    return (x, y, z)


# Unit-cube corners per tier as sorted order-independent lists: blockstate
# rotations map a face onto its own square with reversed winding, so an
# element-wise comparison would wrongly reject rotated opaque cubes.
_UNIT_CORNERS: dict[int, list[tuple[float, float, float]]] = {
    tier: sorted(_corner_tuple(corner) for corner in corners)
    for tier, corners in _FACE_GEOM.items()
}


def _sorted_corners(quad: Any) -> list[tuple[float, float, float]]:
    """Return a quad's corners as a sorted, order-independent list."""
    return sorted(_corner_tuple(point) for point in quad["vtx"])


def _unit_cube_occl(
    quads: np.ndarray, tex_arrays: dict[str, np.ndarray]
) -> bool:
    """Judge the v1 opaque-cube rule: exact unit cube, visible faces opaque."""
    if len(quads) != 6:
        return False
    by_tier: dict[int, Any] = {}
    for quad in quads:
        by_tier[int(quad["shade"])] = quad
    if set(by_tier) != set(_FACE_GEOM):
        return False
    for tier in _FACE_GEOM:
        if _sorted_corners(by_tier[tier]) != _UNIT_CORNERS[tier]:
            return False
    alphas: list[float] = []
    for tier in (TIER_UP, TIER_SOUTH, TIER_EAST):
        array = tex_arrays.get(str(by_tier[tier]["texkey"]))
        if array is None:
            return False
        alphas.append(float(array[..., 3].mean()))
    return float(np.mean(alphas)) >= _OPAQUE_ALPHA


def _compile_elements(
    elements: list[dict[str, Any]],
    textures: dict[str, str],
    ns: str,
    x_rot: int,
    y_rot: int,
    texkey_for: Callable[[dict[str, Any], dict[str, str], str], str | None],
) -> np.ndarray | None:
    """Compile box elements into world-space quads for one variant."""
    rows: list[tuple[Any, Any, str, int]] = []
    kx = (int(x_rot) // 90) % 4
    ky = (int(y_rot) // 90) % 4
    for element in elements:
        if not isinstance(element, dict):
            continue
        bounds = _element_bounds(element)
        faces = element.get("faces")
        if bounds is None or not isinstance(faces, dict):
            continue
        for face_name, tier in _FACE_TIER.items():
            face = faces.get(face_name)
            if not isinstance(face, dict):
                continue
            texkey = texkey_for(face, textures, ns)
            if texkey is None:
                continue
            corners = _box_corners(bounds, tier)
            uvs = _face_uvs(face, tier, corners)
            if uvs is None:
                continue
            world_tier = _FACE_TIER[_rotate(face_name, x_rot, y_rot)]
            rows.append((_rotate_corners(corners, kx, ky), uvs, texkey, world_tier))
    if not rows:
        return None
    quads = np.empty(len(rows), dtype=QUAD_DTYPE)
    for index, (corners, uvs, texkey, shade) in enumerate(rows):
        quads[index] = (corners, uvs, texkey, shade, False)
    keep = is_parallelogram(quads)
    if not keep.any():
        return None
    return quads[keep]


def _element_bounds(
    element: dict[str, Any],
) -> tuple[float, float, float, float, float, float] | None:
    """Parse from/to into six finite floats, or None when malformed."""
    frm = element.get("from")
    to = element.get("to")
    if not isinstance(frm, (list, tuple)) or not isinstance(to, (list, tuple)):
        return None
    if len(frm) != 3 or len(to) != 3:
        return None
    try:
        values = tuple(float(value) for value in (*frm, *to))
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return cast("tuple[float, float, float, float, float, float]", values)


def _box_corners(
    bounds: tuple[float, float, float, float, float, float], tier: int
) -> list[tuple[float, float, float]]:
    """Scale the pinned unit-cube corners of one tier onto an element box."""
    x0, y0, z0, x1, y1, z1 = bounds
    axis = ((x0, x1), (y0, y1), (z0, z1))
    return [
        (axis[0][corner[0] // 16], axis[1][corner[1] // 16], axis[2][corner[2] // 16])
        for corner in _FACE_GEOM[tier]
    ]


def _face_uvs(
    face: dict[str, Any], tier: int, corners: list[tuple[float, float, float]]
) -> list[tuple[float, float]] | None:
    """Resolve per-corner texel coords: explicit uv first, else auto-UV."""
    rect = face.get("uv")
    if rect is None:
        return [_auto_uv(tier, x, y, z) for (x, y, z) in corners]
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    try:
        u1, v1, u2, v2 = (float(value) for value in rect)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (u1, v1, u2, v2)):
        return None
    if min(u1, v1, u2, v2) < 0.0 or max(u1, v1, u2, v2) > 16.0:
        return None
    return [(u1, v1), (u2, v1), (u2, v2), (u1, v2)]


def _auto_uv(tier: int, x: float, y: float, z: float) -> tuple[float, float]:
    """Map one face corner to texel coords per the pinned auto-UV axis table."""
    if tier == TIER_DOWN:
        return (x, 16.0 - z)
    if tier == TIER_UP:
        return (x, z)
    if tier == TIER_NORTH:
        return (16.0 - x, 16.0 - y)
    if tier == TIER_SOUTH:
        return (x, 16.0 - y)
    if tier == TIER_WEST:
        return (16.0 - z, 16.0 - y)
    return (z, 16.0 - y)


def _rot_x_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate one block-space point 90 degrees around the x axis."""
    return (point[0], point[2], 16.0 - point[1])


def _rot_y_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate one block-space point 90 degrees around the y axis."""
    return (16.0 - point[2], point[1], point[0])


def _rotate_corners(
    corners: list[tuple[float, float, float]], kx: int, ky: int
) -> list[tuple[float, float, float]]:
    """Apply x then y blockstate rotation steps to all four corners."""
    out = corners
    for _ in range(kx):
        out = [_rot_x_point(point) for point in out]
    for _ in range(ky):
        out = [_rot_y_point(point) for point in out]
    return out


def _all_variants(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    variants = state.get("variants")
    if isinstance(variants, dict):
        for key, value in variants.items():
            variant = value[0] if isinstance(value, list) and value else value
            if isinstance(variant, dict) and isinstance(variant.get("model"), str):
                out.append((_normalize_state_key(key), variant))
        return out
    multipart = state.get("multipart")
    if isinstance(multipart, list):
        chosen = next(
            (p for p in multipart if isinstance(p, dict) and p.get("when") is None),
            None,
        )
        if chosen is None and multipart and isinstance(multipart[0], dict):
            chosen = multipart[0]
        if chosen is not None:
            apply = chosen.get("apply")
            if isinstance(apply, list):
                apply = apply[0] if apply else None
            if isinstance(apply, dict) and isinstance(apply.get("model"), str):
                out.append(("", apply))
    return out


def _normalize_state_key(key: str) -> str:
    if not key:
        return ""
    pairs = [pair.strip() for pair in key.split(",") if pair.strip()]
    pairs.sort(key=lambda pair: pair.split("=", 1)[0])
    return ",".join(pairs)


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
            return None
        return {"textures": merged_textures, "elements": None}
    return {"textures": merged_textures, "elements": elements}


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


def _rotate(face: str, x: int, y: int) -> str:
    vec = _VEC[face]
    for _ in range((x // 90) % 4):  # y' = z, z' = -y
        vec = (vec[0], vec[2], -vec[1])
    for _ in range((y // 90) % 4):  # x' = -z, z' = x
        vec = (-vec[2], vec[1], vec[0])
    return _NAME[vec]


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


def _block_id(state_name: str) -> str:
    parts = state_name.split("/")
    ns = parts[1]
    stem = parts[-1][: -len(".json")]
    return f"{ns}:{stem}"


def _fallback_blocks(names: set[str]) -> list[tuple[str, str]]:
    """List (namespace, stem) for textures directly under textures/block/."""
    out: list[tuple[str, str]] = []
    for name in sorted(names):
        parts = name.split("/")
        if (
            len(parts) != 5
            or parts[0] != "assets"
            or parts[2] != "textures"
            or parts[3] != "block"
            or not parts[4].endswith(".png")
        ):
            continue
        stem = parts[4][: -len(".png")]
        if not stem or not all(
            c.islower() or c.isdigit() or c == "_" for c in stem
        ):
            continue
        out.append((parts[1], stem))
    return out


def _tint_for(block_id: str, state_key: str) -> tuple[int, int, int] | None:
    path = block_id.split(":", 1)[-1]
    if path == "water" or path == "bubble_column":
        return _TINTS["water"]
    if path == "redstone_wire":
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
    factor = power / 15.0
    return tuple(
        round(low + (high - low) * factor)
        for low, high in zip(_REDSTONE_LOW, _REDSTONE_HIGH)
    )  # type: ignore[return-value]


def _apply_tint(array: np.ndarray, tint: tuple[int, int, int]) -> np.ndarray:
    factor = np.array(tint + (255,), dtype=np.uint16)
    return ((array.astype(np.uint16) * factor) // 255).astype(np.uint8)


def _load_texture(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    tex_ref: str,
    tex_key: str,
    cache: dict[str, np.ndarray | None],
) -> np.ndarray | None:
    if tex_key in cache:
        return cache[tex_key]
    file = _asset_path(ns, tex_ref, "textures", ".png")
    result: np.ndarray | None = None
    if file is not None and file in names:
        try:
            image = _decode_png(archive.read(file))
            result = np.asarray(image, dtype=np.uint8)
        except RenderError:
            raise
        except Exception:
            result = None
    cache[tex_key] = result
    return result


def _decode_png(data: bytes) -> Any:
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGBA")
    width, height = image.size
    if width != height:  # animation strip: first frame
        side = min(width, height)
        image = image.crop((0, 0, side, side))
    if image.size != (_TEX_SIZE, _TEX_SIZE):
        image = image.resize((_TEX_SIZE, _TEX_SIZE), Image.Resampling.BOX)
    return image


def _write_npz(
    dest: Path,
    tex_arrays: dict[str, np.ndarray],
    quad_map: dict[str, np.ndarray],
    occl_map: dict[str, bool],
    pack_mtime: float,
) -> None:
    entries: dict[str, np.ndarray] = {
        "meta/version": np.array(_NPZ_VERSION),
        "meta/pack_mtime": np.array(pack_mtime),
    }
    for array_key, array in tex_arrays.items():
        entries[f"tex/{array_key}"] = array
    for key, quads in quad_map.items():
        entries[f"quads/{key}"] = sort_quads(quads)
    for key, occl in occl_map.items():
        entries[f"occl/{key}"] = np.array(occl, dtype=np.uint8)
    try:
        out = zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED)
    except OSError as exc:
        raise _CacheWriteError(
            f"extract: cannot write npz: {dest}: {exc}"
        ) from exc
    with out:
        for key in sorted(entries):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, entries[key], allow_pickle=False)
            info = zipfile.ZipInfo(key, date_time=_FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            out.writestr(info, buffer.getvalue())
