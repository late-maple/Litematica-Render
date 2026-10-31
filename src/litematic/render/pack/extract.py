"""Pack extraction orchestration: blockstates → quads → npz."""

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.pack.blockstate import _block_id
from litematic.render.pack.compile import _extract_block, _model_quads_key
from litematic.render.pack.elements import _unit_cube_occl
from litematic.render.pack.resolve import FRAME_PAD_KEY, _tint_for
from litematic.render.pack.stack import PackStack, open_stack
from litematic.render.pack.texture import (
    _Budget,
    _cached_model,
    _load_texture,
    _register_array,
    _register_wire_dust,
    _resolve_ref,
    _texture_key,
)
from litematic.render.quads import sort_quads, unit_cube_quads
from litematic.render.shapes import builtin_quads


class ExtractSummary(NamedTuple):
    """Outcome of one pack extraction."""

    path: Path
    blocks: int
    skipped: int



class _CacheWriteError(RenderError):
    """Raised when the npz destination cannot be written."""



_FIXED_DATE = (1980, 1, 1, 0, 0, 0)



_NPZ_VERSION = "pack-quads.1"

_FINGERPRINT: list[str] = []



def _code_fingerprint() -> str:
    """Digest of the extraction-side sources; any change forces a rebuild."""
    if not _FINGERPRINT:
        root = Path(__file__).parent
        paths = sorted(root.glob("*.py"))
        paths += sorted((root.parent / "shapes").glob("*.py"))
        paths.append(root.parent / "quads.py")
        digest = hashlib.sha256()
        try:
            for path in paths:
                digest.update(str(path).encode("utf-8"))
                digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        except OSError as exc:
            raise RenderError(
                f"extract: cannot fingerprint extraction code: {exc}"
            ) from exc
        _FINGERPRINT.append(digest.hexdigest()[:16])
    return _FINGERPRINT[0]



def _require_pillow() -> None:
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise RenderError(
            "extract: Pillow is required — pip install litematic-render[pack]"
        ) from exc



def _extract_pack(
    sources: str | Path | Sequence[str | Path],
    dest: str | Path,
    pack_mtime: float | tuple[float, ...] | None = None,
) -> ExtractSummary:
    if isinstance(sources, (str, Path)):
        sources = [sources]
    src = Path(sources[0])
    if not src.is_file():
        raise RenderError(f"extract: not a resource pack: {src}: no such file")
    if pack_mtime is None:
        try:
            pack_mtime = src.stat().st_mtime
        except OSError as exc:
            raise RenderError(f"extract: not a resource pack: {src}: {exc}") from exc
    with open_stack(sources) as stack:
        names = stack.names()
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
        budget = _Budget()
        model_cache: dict[str, dict[str, Any] | None] = {}
        mquad_cache: dict[str, str | None] = {}
        tex_arrays: dict[str, np.ndarray] = {}
        quad_map: dict[str, np.ndarray] = {}
        occl_map: dict[str, bool] = {}
        parts_map: dict[str, str] = {}
        mquad_map: dict[str, np.ndarray] = {}
        blocks = 0
        skipped = 0
        covered: set[str] = set()
        for state_name in state_files:
            block_id = _block_id(state_name)
            try:
                states, parts_json = _extract_block(
                    stack,
                    state_name,
                    block_id,
                    model_cache,
                    texture_cache,
                    tex_arrays,
                    mquad_cache,
                    mquad_map,
                    budget,
                )
            except RenderError:
                raise
            except Exception:
                skipped += 1
                continue
            if not states and parts_json is None:
                skipped += 1
                continue
            covered.add(block_id)
            for state_key, (quads, occl) in states.items():
                key = f"{block_id}[{state_key}]"
                quad_map[key] = quads
                occl_map[key] = occl
            if parts_json is not None:
                parts_map[f"parts/{block_id}"] = parts_json
            blocks += 1
        items_map = _extract_items(
            stack, texture_cache, tex_arrays, model_cache, budget
        )
        _extract_frame_pad(
            stack, texture_cache, tex_arrays, model_cache,
            mquad_cache, mquad_map, budget,
        )
        _register_direct_items(stack, texture_cache, tex_arrays, budget)
        for ns, stem in _fallback_blocks(names):
            block_id = f"{ns}:{stem}"
            if block_id in covered:
                # Blocks with pack quads never fall through to the layers
                # below; their unmatched states stay for magenta.
                continue
            array_key = _fallback_array_key(
                stack, ns, stem, texture_cache, tex_arrays, budget
            )
            if array_key is None:
                continue
            if builtin_quads(block_id, {}) is not None:
                # Builtin families never get extraction-side geometry: it
                # would shadow compose's state-aware builtin step. Their
                # texture is registered under the canonical block key (raw
                # plus tint variant above) for compose to backfill.
                continue
            # Spec 5.5 texture-direct cube: the last layer before magenta
            # for blocks the builtin table does not know.
            cube = unit_cube_quads(array_key)
            quad_map[f"{block_id}[]"] = cube
            occl_map[f"{block_id}[]"] = _unit_cube_occl(cube, tex_arrays)
            blocks += 1
        asset_ns = {
            name.split("/")[1]
            for name in names
            if name.startswith("assets/") and len(name.split("/")) > 2
        }
        for ns in sorted(asset_ns):
            _register_wire_dust(stack, ns, texture_cache, tex_arrays, budget)
    _write_npz(
        Path(dest), tex_arrays, quad_map, occl_map, parts_map, mquad_map,
        pack_mtime, items_map,
    )
    return ExtractSummary(Path(dest), blocks, skipped)



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



def _fallback_array_key(
    stack: PackStack,
    ns: str,
    stem: str,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    budget: _Budget,
) -> str | None:
    """Resolve a textures/block/<stem>.png fallback into an array key."""
    tex_ref = f"block/{stem}"
    tex_key = _texture_key(tex_ref, ns)
    raw = _load_texture(stack, ns, tex_ref, tex_key, texture_cache, budget)
    if raw is None:
        return None
    tint = _tint_for(f"{ns}:{stem}", "")
    return _register_array(raw, tex_key, tint, tex_arrays)



_ARMORSTAND_TEXKEY = "entity/armorstand/armorstand"



def _extract_frame_pad(
    stack: PackStack,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    model_cache: dict[str, dict[str, Any] | None],
    mquad_cache: dict[str, str | None],
    mquad_out: dict[str, np.ndarray],
    budget: _Budget,
) -> None:
    """Compile the vanilla item-frame model once for compose to rotate."""
    subkey = _model_quads_key(
        stack, "minecraft", "minecraft:block/item_frame", 0, 0, False,
        "minecraft:item_frame", model_cache, texture_cache, tex_arrays,
        mquad_cache, mquad_out, budget,
    )
    if subkey is not None:
        mquad_out.setdefault(FRAME_PAD_KEY, mquad_out[subkey])



def _extract_items(
    stack: PackStack,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    model_cache: dict[str, dict[str, Any] | None],
    budget: _Budget,
) -> dict[str, dict[str, str]]:
    """Classify 1.21.9 item assets: block models or generated sprites.

    Two sources: items/<id>.json definitions (block items point straight
    at models/block/) and models/item/<id>.json chains ending at
    builtin/generated (flat sprites). Unresolvable or non-model types
    stay unrecorded.
    """
    items: dict[str, dict[str, str]] = {}
    for path in sorted(stack.names()):
        if not path.startswith("assets/") or not path.endswith(".json"):
            continue
        parts = path.split("/")
        ns, stem = parts[1], parts[-1][: -len(".json")]
        if len(parts) == 4 and parts[2] == "items":
            data = _read_json(stack, path)
            model = data.get("model") if isinstance(data, dict) else None
            if not isinstance(model, dict) or model.get("type") != "minecraft:model":
                continue
            ref = model.get("model")
            if isinstance(ref, str) and ref:
                _classify_item_ref(
                    ns, ref, f"{ns}:{stem}", items, stack,
                    texture_cache, tex_arrays, model_cache, budget,
                )
        elif len(parts) == 5 and parts[2] == "models" and parts[3] == "item":
            _classify_item_ref(
                ns, f"{ns}:item/{stem}", f"{ns}:{stem}", items, stack,
                texture_cache, tex_arrays, model_cache, budget,
            )
    raw = _load_texture(
        stack, "minecraft", _ARMORSTAND_TEXKEY, _ARMORSTAND_TEXKEY,
        texture_cache, budget,
    )
    if raw is not None:
        tex_arrays.setdefault(_ARMORSTAND_TEXKEY, raw)
    return items


def _classify_item_ref(
    ns: str,
    ref: str,
    item_id: str,
    items: dict[str, dict[str, str]],
    stack: PackStack,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    model_cache: dict[str, dict[str, Any] | None],
    budget: _Budget,
) -> None:
    """Record one item as block-model or flat sprite, or leave it out."""
    if item_id in items:
        return
    model = _cached_model(stack, ns, ref, model_cache)
    if model is None:
        return
    if model["elements"] is not None:
        block_id = _block_id_of_ref(ns, ref)
        if block_id is not None:
            items[item_id] = {"block": block_id}
        return
    if not model.get("builtin_cut"):
        return
    textures = model["textures"]
    layer0 = _resolve_ref(textures.get("layer0"), textures)
    if layer0 is None:
        return
    tex_key = _texture_key(layer0, ns)
    raw = _load_texture(stack, ns, layer0, tex_key, texture_cache, budget)
    if raw is None:
        return
    tint = _tint_for(item_id, "")
    items[item_id] = {"flat": _register_array(raw, tex_key, tint, tex_arrays)}


def _block_id_of_ref(ns: str, ref: str) -> str | None:
    """Map a model ref like minecraft:block/stone to the block id."""
    ref_ns, _, path = ref.partition(":")
    if not path:
        ref_ns, path = ns, ref_ns
    if not path.startswith("block/"):
        return None
    stem = path[len("block/") :]
    if not stem or "/" in stem:
        return None
    return f"{ref_ns}:{stem}"


def _register_direct_items(
    stack: PackStack,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    budget: _Budget,
) -> None:
    """Register textures/item/*.png so texture-only packs render entities.

    Vanilla packs already cover these through the model walk; the
    setdefault only fills the gaps (texture-only add-ons, unmodeled items).
    """
    for path in sorted(stack.names()):
        parts = path.split("/")
        if (
            len(parts) != 5
            or parts[0] != "assets"
            or parts[2] != "textures"
            or parts[3] != "item"
            or not parts[4].endswith(".png")
        ):
            continue
        stem = parts[4][: -len(".png")]
        if not stem or "/" in stem:
            continue
        tex_key = f"item/{stem}"
        if tex_key in tex_arrays:
            continue
        raw = _load_texture(
            stack, parts[1], f"item/{stem}", tex_key, texture_cache, budget
        )
        if raw is not None:
            tex_arrays.setdefault(tex_key, raw)


def _read_json(stack: PackStack, path: str) -> Any:
    raw = stack.read(path)
    if raw is None:  # unreachable: the path came from the stack enumeration
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _write_npz(
    dest: Path,
    tex_arrays: dict[str, np.ndarray],
    quad_map: dict[str, np.ndarray],
    occl_map: dict[str, bool],
    parts_map: dict[str, str],
    mquad_map: dict[str, np.ndarray],
    pack_mtime: float | tuple[float, ...],
    items_map: dict[str, dict[str, str]],
) -> None:
    entries: dict[str, np.ndarray] = {
        "meta/version": np.array(_NPZ_VERSION),
        "meta/pack_mtime": np.array(pack_mtime),
        "meta/code_hash": np.array(_code_fingerprint()),
        "items": np.array(json.dumps(items_map)),
    }
    for array_key, array in tex_arrays.items():
        entries[f"tex/{array_key}"] = array
    for key, quads in quad_map.items():
        entries[f"quads/{key}"] = sort_quads(quads)
    for key, occl in occl_map.items():
        entries[f"occl/{key}"] = np.array(occl, dtype=np.uint8)
    for key, record in parts_map.items():
        entries[key] = np.array(record)
    for key, quads in mquad_map.items():
        entries[f"mquads/{key}"] = sort_quads(quads)
    try:
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as out:
            for key in sorted(entries):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, entries[key], allow_pickle=False)
                info = zipfile.ZipInfo(key, date_time=_FIXED_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                out.writestr(info, buffer.getvalue())
    except OSError as exc:
        raise _CacheWriteError(
            f"extract: cannot write npz: {dest}: {exc}"
        ) from exc
