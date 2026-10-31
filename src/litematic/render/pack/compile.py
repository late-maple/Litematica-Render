"""Per-block state compilation: variants, multipart parts, chests, texkeys."""

import json
from typing import Any

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.pack.blockstate import _parse_blockstate
from litematic.render.pack.elements import (
    _compile_elements,
    _unit_cube_occl,
)
from litematic.render.pack.entities import chest_quads
from litematic.render.pack.resolve import _tint_for
from litematic.render.pack.stack import PackStack
from litematic.render.pack.texture import (
    _apply_tint,
    _Budget,
    _cached_model,
    _load_texture,
    _register_array,
    _register_builtin_texture,
    _resolve_ref,
    _texture_key,
)
from litematic.render.quads import _redstone_tint, sort_quads, unit_cube_quads

_QUAD_FUSE = 256



def _extract_block(
    stack: PackStack,
    state_name: str,
    block_id: str,
    model_cache: dict[str, dict[str, Any] | None],
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    mquad_cache: dict[str, str | None],
    mquad_out: dict[str, np.ndarray],
    budget: _Budget,
) -> tuple[dict[str, tuple[np.ndarray, bool]], str | None]:
    """Extract variant quads plus the block's multipart part records."""
    raw = stack.read(state_name)
    if raw is None:  # unreachable: the path came from the stack enumeration
        raise RenderError(f"extract: missing blockstate: {state_name}")
    state = json.loads(raw)
    variants, parts, model_refs = _parse_blockstate(state)
    ns = state_name.split("/")[1]
    result: dict[str, tuple[np.ndarray, bool]] = {}
    for state_key, applies in variants:
        quads: list[np.ndarray] = []
        for variant in applies:
            x_rot = int(variant.get("x", 0))
            y_rot = int(variant.get("y", 0))
            uvlock = bool(variant.get("uvlock", False))
            model = _cached_model(stack, ns, variant["model"], model_cache)
            if model is None:
                continue
            compiled = _variant_quads(
                stack,
                ns,
                model,
                block_id,
                state_key,
                x_rot,
                y_rot,
                uvlock,
                texture_cache,
                tex_arrays,
                budget,
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
        stack,
        ns,
        block_id,
        parts,
        model_cache,
        texture_cache,
        tex_arrays,
        mquad_cache,
        mquad_out,
        budget,
    )
    expanded = _chest_states(stack, ns, block_id, texture_cache, tex_arrays, budget)
    if expanded:
        # The expanded facing/type states supersede variant stubs: a stub
        # key with more property pairs would otherwise out-specify them.
        result = expanded
    if not result and parts_json is None:
        # No pack geometry (missing models, builtin-cut chains): resolve
        # the builtin layer's texture per spec 5.3 so compose's state-aware
        # builtin step can backfill it under the canonical block key.
        _register_builtin_texture(
            stack, ns, block_id, model_refs, model_cache,
            texture_cache, tex_arrays, budget,
        )
    return result, parts_json



def _extract_parts(
    stack: PackStack,
    ns: str,
    block_id: str,
    parts: list[dict[str, Any]] | None,
    model_cache: dict[str, dict[str, Any] | None],
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    mquad_cache: dict[str, str | None],
    mquad_out: dict[str, np.ndarray],
    budget: _Budget,
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
            stack,
            ns,
            part["model"],
            part["x"],
            part["y"],
            part["uvlock"],
            block_id,
            model_cache,
            texture_cache,
            tex_arrays,
            mquad_cache,
            mquad_out,
            budget,
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
                tint = _redstone_tint(power)
                tex_arrays.setdefault(
                    f"{base}@{tint[0]},{tint[1]},{tint[2]}",
                    _apply_tint(raw, tint),
                )
    return json.dumps({"parts": records})


def _model_quads_key(
    stack: PackStack,
    ns: str,
    ref: str,
    x_rot: int,
    y_rot: int,
    uvlock: bool,
    block_id: str,
    model_cache: dict[str, dict[str, Any] | None],
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    mquad_cache: dict[str, str | None],
    mquad_out: dict[str, np.ndarray],
    budget: _Budget,
) -> str | None:
    """Compile one part model's quads once, keyed by (ns, ref, rotation)."""
    cache_key = f"{ns}|{ref}|x{x_rot}|y{y_rot}|u{int(uvlock)}"
    if cache_key in mquad_cache:
        return mquad_cache[cache_key]
    subkey = cache_key
    model = _cached_model(stack, ns, ref, model_cache)
    quads: np.ndarray | None = None
    if model is not None and not model.get("builtin_cut"):
        if model["elements"] is not None:

            def texkey_for(
                face: dict[str, Any], textures: dict[str, str], face_ns: str
            ) -> str | None:
                return _element_texkey(
                    stack,
                    face_ns,
                    face,
                    textures,
                    block_id,
                    "",
                    texture_cache,
                    tex_arrays,
                    budget,
                )

            quads = _compile_elements(
                model["elements"],
                model["textures"],
                ns,
                x_rot,
                y_rot,
                texkey_for,
                uvlock=uvlock,
            )
        else:
            quads = _stub_quads(
                stack,
                ns,
                model,
                block_id,
                "",
                texture_cache,
                tex_arrays,
                budget,
            )
    if quads is None or not len(quads):
        mquad_cache[cache_key] = None
        return None
    mquad_out[subkey] = sort_quads(quads)
    mquad_cache[cache_key] = subkey
    return subkey



def _variant_quads(
    stack: PackStack,
    ns: str,
    model: dict[str, Any],
    block_id: str,
    state_key: str,
    x_rot: int,
    y_rot: int,
    uvlock: bool,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    budget: _Budget,
) -> np.ndarray | None:
    """Compile one variant's quads: element boxes, else a stub unit cube."""
    if model["elements"] is None:
        if model.get("builtin_cut"):
            # Chains terminated by a builtin/* parent (chests, heads) have
            # no pack geometry: defer to the builtin shape table.
            return None
        return _stub_quads(
            stack,
            ns,
            model,
            block_id,
            state_key,
            texture_cache,
            tex_arrays,
            budget,
        )

    def texkey_for(
        face: dict[str, Any], textures: dict[str, str], face_ns: str
    ) -> str | None:
        return _element_texkey(
            stack,
            face_ns,
            face,
            textures,
            block_id,
            state_key,
            texture_cache,
            tex_arrays,
            budget,
        )

    return _compile_elements(
        model["elements"],
        model["textures"],
        ns,
        x_rot,
        y_rot,
        texkey_for,
        uvlock=uvlock,
    )



_CHEST_STATE_KEYS = [
    f"facing={facing},type={kind}"
    for facing in ("north", "south", "east", "west")
    for kind in ("single", "left", "right")
]

# Vanilla ender chests have no type property: expand facing only.
_ENDER_CHEST_STATE_KEYS = [
    f"facing={facing}" for facing in ("north", "south", "east", "west")
]



def _chest_states(
    stack: PackStack,
    ns: str,
    block_id: str,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    budget: _Budget,
) -> dict[str, tuple[np.ndarray, bool]]:
    """Expand an element-less chest-family block into facing/type states.

    Vanilla ships chest geometry only through the block-entity renderer:
    the blockstate never enumerates facing/type variants (a bare empty
    variant at best), so the entity textures carry the model and the state
    domain is expanded here.
    """

    def resolve(tex_ref: str) -> str | None:
        tex_key = _texture_key(tex_ref, ns)
        raw = _load_texture(stack, ns, tex_ref, tex_key, texture_cache, budget)
        if raw is None:
            return None
        return _register_array(raw, tex_key, None, tex_arrays)

    if chest_quads(block_id, "", resolve) is None:
        return {}
    stem = block_id.partition(":")[2]
    keys = _ENDER_CHEST_STATE_KEYS if stem == "ender_chest" else _CHEST_STATE_KEYS
    states: dict[str, tuple[np.ndarray, bool]] = {}
    for key in keys:
        quads = chest_quads(block_id, key, resolve)
        if quads is not None:
            states[key] = (quads, False)
    return states



def _element_texkey(
    stack: PackStack,
    ns: str,
    face: dict[str, Any],
    textures: dict[str, str],
    block_id: str,
    state_key: str,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    budget: _Budget,
) -> str | None:
    """Resolve one element face's texture ref into a tex array key."""
    tex_ref = _resolve_ref(face.get("texture"), textures)
    if tex_ref is None:
        return None
    return _ref_array_key(
        stack,
        ns,
        tex_ref,
        "tintindex" in face,
        block_id,
        state_key,
        texture_cache,
        tex_arrays,
        budget,
    )



def _ref_array_key(
    stack: PackStack,
    ns: str,
    tex_ref: str,
    tinted: bool,
    block_id: str,
    state_key: str,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    budget: _Budget,
) -> str | None:
    """Load a texture ref and register its raw or tinted array key."""
    tex_key = _texture_key(tex_ref, ns)
    raw = _load_texture(stack, ns, tex_ref, tex_key, texture_cache, budget)
    if raw is None:
        return None
    tint = _tint_for(block_id, state_key) if tinted else None
    return _register_array(raw, tex_key, tint, tex_arrays)



def _stub_quads(
    stack: PackStack,
    ns: str,
    model: dict[str, Any],
    block_id: str,
    state_key: str,
    texture_cache: dict[str, np.ndarray | None],
    tex_arrays: dict[str, np.ndarray],
    budget: _Budget,
) -> np.ndarray | None:
    """Build unit-cube quads from an element-less model's first texture."""
    array_key = _ref_array_key(
        stack,
        ns,
        model["stub_ref"],
        True,
        block_id,
        state_key,
        texture_cache,
        tex_arrays,
        budget,
    )
    if array_key is None:
        return None
    return unit_cube_quads(array_key)
