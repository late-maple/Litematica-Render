"""Resource pack extraction: blockstates → models → textures → npz."""

import io
import json
import zipfile
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from litematic.render.errors import RenderError

__all__ = ["ExtractSummary"]

_TEX_SIZE = 16
_FIXED_DATE = (1980, 1, 1, 0, 0, 0)
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
_WORLD_FACES = ("up", "east", "south")
_ALL_FACES = tuple(_VEC)


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
        maps: dict[str, str] = {}
        blocks = 0
        skipped = 0
        covered: set[str] = set()
        for state_name in state_files:
            try:
                state_maps = _extract_block(
                    archive, names, state_name, model_cache, texture_cache, tex_arrays
                )
            except RenderError:
                raise
            except Exception:
                skipped += 1
                continue
            if not state_maps:
                skipped += 1
                continue
            block_id = _block_id(state_name)
            covered.add(block_id)
            for state_key, faces in state_maps.items():
                for face, array_key in faces.items():
                    maps[f"map/{block_id}[{state_key}]/{face}"] = array_key
            blocks += 1
        for ns, stem in _fallback_blocks(names):
            block_id = f"{ns}:{stem}"
            if block_id in covered:
                continue
            tex_ref = f"block/{stem}"
            tex_key = _texture_key(tex_ref, ns)
            raw = _load_texture(archive, names, ns, tex_ref, tex_key, texture_cache)
            if raw is None:
                continue
            tint = _tint_for(block_id, "")
            if tint is None:
                tex_arrays.setdefault(tex_key, raw)
                array_key = tex_key
            else:
                array_key = f"{tex_key}@{tint[0]},{tint[1]},{tint[2]}"
                if array_key not in tex_arrays:
                    tex_arrays[array_key] = _apply_tint(raw, tint)
            for face in _ALL_FACES:
                maps[f"map/{block_id}[]/{face}"] = array_key
            blocks += 1
    _write_npz(Path(dest), tex_arrays, maps, pack_mtime)
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
) -> dict[str, dict[str, str]]:
    """Extract every variant of one blockstate file, keyed by state."""
    state = json.loads(archive.read(state_name))
    variants = _all_variants(state)
    if not variants:
        return {}
    ns = state_name.split("/")[1]
    block_id = _block_id(state_name)
    result: dict[str, dict[str, str]] = {}
    for state_key, variant in variants:
        x_rot = int(variant.get("x", 0))
        y_rot = int(variant.get("y", 0))
        model = _cached_model(archive, names, ns, variant["model"], model_cache)
        if model is None:
            continue
        faces: dict[str, str] = {}
        for world_face in _ALL_FACES:
            model_face = _source_face(world_face, x_rot, y_rot)
            array_key = _face_array_key(
                archive,
                names,
                ns,
                model,
                model_face,
                block_id,
                state_key,
                texture_cache,
                tex_arrays,
            )
            if array_key is not None:
                faces[world_face] = array_key
        if all(required in faces for required in _WORLD_FACES):
            result[state_key] = faces
    return result


def _face_array_key(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    model: dict[str, Any],
    model_face: str,
    block_id: str,
    state_key: str,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
) -> str | None:
    face = model["faces"].get(model_face)
    if not isinstance(face, dict):
        return None
    tex_ref = _resolve_ref(face.get("texture"), model["textures"])
    if tex_ref is None:
        return None
    tex_key = _texture_key(tex_ref, ns)
    raw = _load_texture(archive, names, ns, tex_ref, tex_key, texture_cache)
    if raw is None:
        return None
    tint = _tint_for(block_id, state_key) if "tintindex" in face else None
    if tint is None:
        tex_arrays.setdefault(tex_key, raw)
        return tex_key
    array_key = f"{tex_key}@{tint[0]},{tint[1]},{tint[2]}"
    if array_key not in tex_arrays:
        tex_arrays[array_key] = _apply_tint(raw, tint)
    return array_key


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
        if model is not None:
            model["faces"] = _face_textures(model)
        cache[key] = model
    return cache[key]


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
        return None
    return {"textures": merged_textures, "elements": elements}


def _face_textures(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    faces: dict[str, dict[str, Any]] = {}
    for element in model["elements"]:
        if not isinstance(element, dict):
            continue
        element_faces = element.get("faces")
        if not isinstance(element_faces, dict):
            continue
        for face_name, face in element_faces.items():
            if isinstance(face, dict) and face_name not in faces:
                faces[face_name] = face
    return faces


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
    for _ in range((x // 90) % 4):  # y' = -z, z' = y
        vec = (vec[0], -vec[2], vec[1])
    for _ in range((y // 90) % 4):  # x' = -z, z' = x
        vec = (-vec[2], vec[1], vec[0])
    return _NAME[vec]


def _source_face(world_face: str, x: int, y: int) -> str:
    for face in _ALL_FACES:
        if _rotate(face, x, y) == world_face:
            return face
    raise AssertionError("unreachable")


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
    maps: dict[str, str],
    pack_mtime: float,
) -> None:
    entries: dict[str, np.ndarray] = {
        "meta/version": np.array("pack"),
        "meta/pack_mtime": np.array(pack_mtime),
    }
    for array_key, array in tex_arrays.items():
        entries[f"tex/{array_key}"] = array
    for map_key, array_key in maps.items():
        entries[map_key] = np.array(array_key)
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
