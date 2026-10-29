"""Pack extraction orchestration: blockstates → quads → npz."""

import io
import json
import zipfile
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.pack.blockstate import _block_id, _parse_blockstate
from litematic.render.pack.elements import (
    _compile_elements,
    _unit_cube_occl,
)
from litematic.render.pack.texture import (
    _apply_tint,
    _cached_model,
    _load_texture,
    _register_builtin_texture,
    _register_wire_dust,
    _resolve_ref,
    _texture_key,
    _tint_for,
)
from litematic.render.quads import (
    redstone_tint,
    sort_quads,
    unit_cube_quads,
)
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



_QUAD_FUSE = 256



def _require_pillow() -> None:
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise RenderError(
            "extract: Pillow is required — pip install litematic-render[pack]"
        ) from exc



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
                    archive,
                    names,
                    state_name,
                    block_id,
                    model_cache,
                    texture_cache,
                    tex_arrays,
                    mquad_cache,
                    mquad_map,
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
        for ns, stem in _fallback_blocks(names):
            block_id = f"{ns}:{stem}"
            if block_id in covered:
                # Blocks with pack quads never fall through to the layers
                # below; their unmatched states stay for magenta.
                continue
            array_key = _fallback_array_key(
                archive, names, ns, stem, texture_cache, tex_arrays
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
            _register_wire_dust(archive, names, ns, texture_cache, tex_arrays)
    _write_npz(
        Path(dest), tex_arrays, quad_map, occl_map, parts_map, mquad_map,
        pack_mtime,
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



def _extract_block(
    archive: zipfile.ZipFile,
    names: set[str],
    state_name: str,
    block_id: str,
    model_cache: dict[str, dict[str, Any] | None],
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    mquad_cache: dict[str, str | None],
    mquad_out: dict[str, np.ndarray],
) -> tuple[dict[str, tuple[np.ndarray, bool]], str | None]:
    """Extract variant quads plus the block's multipart part records."""
    state = json.loads(archive.read(state_name))
    variants, parts, model_refs = _parse_blockstate(state)
    ns = state_name.split("/")[1]
    result: dict[str, tuple[np.ndarray, bool]] = {}
    for state_key, applies in variants:
        quads: list[np.ndarray] = []
        for variant in applies:
            x_rot = int(variant.get("x", 0))
            y_rot = int(variant.get("y", 0))
            model = _cached_model(archive, names, ns, variant["model"], model_cache)
            if model is None:
                continue
            compiled = _variant_quads(
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
            if compiled is not None:
                quads.append(compiled)
        if not quads:
            continue
        merged = quads[0] if len(quads) == 1 else np.concatenate(quads)
        if len(merged) > _QUAD_FUSE:
            continue
        result[state_key] = (merged, _unit_cube_occl(merged, tex_arrays))
    parts_json = _extract_parts(
        archive,
        names,
        ns,
        block_id,
        parts,
        model_cache,
        texture_cache,
        tex_arrays,
        mquad_cache,
        mquad_out,
    )
    if not result and parts_json is None:
        # No pack geometry (missing models, builtin-cut chains): resolve
        # the builtin layer's texture per spec 5.3 so compose's state-aware
        # builtin step can backfill it under the canonical block key.
        _register_builtin_texture(
            archive, names, ns, block_id, model_refs, model_cache,
            texture_cache, tex_arrays,
        )
    return result, parts_json


def _extract_parts(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    block_id: str,
    parts: list[dict[str, Any]] | None,
    model_cache: dict[str, dict[str, Any] | None],
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    mquad_cache: dict[str, str | None],
    mquad_out: dict[str, np.ndarray],
) -> str | None:
    """Compile multipart part records into lazily evaluated npz entries.

    Each part's geometry compiles once per (model, rotation) and the raw
    condition tree ships alongside it; render-time evaluation against the
    live properties replaces key enumeration entirely.
    """
    if parts is None:
        return None
    records: list[dict[str, Any]] = []
    for part in parts:
        subkey = _model_quads_key(
            archive,
            names,
            ns,
            part["model"],
            part["x"],
            part["y"],
            block_id,
            model_cache,
            texture_cache,
            tex_arrays,
            mquad_cache,
            mquad_out,
        )
        if subkey is not None:
            records.append({"when": part["when"], "quads": subkey})
    if not records:
        return None
    if block_id.partition(":")[2] == "redstone_wire":
        # Part quads ship untinted (the dust tint depends on the live
        # power); register every power variant of their textures so the
        # lookup-side texkey rewrite always finds the tinted array.
        bases = {
            str(key).partition("@")[0]
            for subkey in (r["quads"] for r in records)
            for key in mquad_out[subkey]["texkey"]
        }
        for base in sorted(bases):
            raw = tex_arrays.get(base)
            if raw is None:
                continue
            for power in range(16):
                tint = redstone_tint(power)
                tex_arrays.setdefault(
                    f"{base}@{tint[0]},{tint[1]},{tint[2]}",
                    _apply_tint(raw, tint),
                )
    return json.dumps({"parts": records})


def _model_quads_key(
    archive: zipfile.ZipFile,
    names: set[str],
    ns: str,
    ref: str,
    x_rot: int,
    y_rot: int,
    block_id: str,
    model_cache: dict[str, dict[str, Any] | None],
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    mquad_cache: dict[str, str | None],
    mquad_out: dict[str, np.ndarray],
) -> str | None:
    """Compile one part model's quads once, keyed by (ns, ref, rotation)."""
    cache_key = f"{ns}|{ref}|x{x_rot}|y{y_rot}"
    if cache_key in mquad_cache:
        return mquad_cache[cache_key]
    subkey = cache_key
    model = _cached_model(archive, names, ns, ref, model_cache)
    quads: np.ndarray | None = None
    if model is not None and not model.get("builtin_cut"):
        if model["elements"] is not None:

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
                    "",
                    texture_cache,
                    tex_arrays,
                )

            quads = _compile_elements(
                model["elements"],
                model["textures"],
                ns,
                x_rot,
                y_rot,
                texkey_for,
            )
        else:
            quads = _stub_quads(
                archive, names, ns, model, block_id, "", texture_cache, tex_arrays
            )
    if quads is None or not len(quads):
        mquad_cache[cache_key] = None
        return None
    mquad_out[subkey] = sort_quads(quads)
    mquad_cache[cache_key] = subkey
    return subkey



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
        if model.get("builtin_cut"):
            # Chains terminated by a builtin/* parent (chests, heads) have
            # no pack geometry: defer to the builtin shape table.
            return None
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



def _write_npz(
    dest: Path,
    tex_arrays: dict[str, np.ndarray],
    quad_map: dict[str, np.ndarray],
    occl_map: dict[str, bool],
    parts_map: dict[str, str],
    mquad_map: dict[str, np.ndarray],
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
    for key, record in parts_map.items():
        entries[key] = np.array(record)
    for key, quads in mquad_map.items():
        entries[f"mquads/{key}"] = sort_quads(quads)
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
