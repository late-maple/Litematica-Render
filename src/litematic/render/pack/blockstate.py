"""Blockstate JSON parsing: variants and multipart part records."""

import json
from collections.abc import Mapping
from typing import Any


def _normalize_state_key(key: str) -> str:
    if not key:
        return ""
    pairs = [pair.strip() for pair in key.split(",") if pair.strip()]
    pairs.sort(key=lambda pair: pair.split("=", 1)[0])
    return ",".join(pairs)



def _block_id(state_name: str) -> str:
    parts = state_name.split("/")
    ns = parts[1]
    stem = parts[-1][: -len(".json")]
    return f"{ns}:{stem}"



def _state_pairs(state_key: str) -> dict[str, str]:
    """Parse a state key into its property pairs; the empty key is {}."""
    pairs: dict[str, str] = {}
    for part in state_key.split(","):
        prop, sep, value = part.partition("=")
        if sep:
            pairs[prop] = value
    return pairs



def _when_matches(
    condition: Mapping[str, Any], values: Mapping[str, str]
) -> bool:
    """Vanilla multipart condition: OR/AND nesting, | alternatives."""
    if not isinstance(condition, Mapping):
        return False
    for key, expected in condition.items():
        if key == "OR" or key == "AND":
            if not isinstance(expected, list):
                return False
            sub = [_when_matches(item, values) for item in expected]
            if key == "OR" and not any(sub):
                return False
            if key == "AND" and not all(sub):
                return False
        else:
            actual = values.get(str(key))
            if actual is None or actual not in str(expected).split("|"):
                return False
    return True



def _parse_blockstate(
    state: dict[str, Any],
) -> tuple[
    list[tuple[str, list[dict[str, Any]]]],
    list[dict[str, Any]] | None,
    list[str],
]:
    """Split a blockstate into variant entries and multipart part records.

    Variant keys are alternatives and map to one apply, keyed for the
    state-keyed section. Conditional multiparts keep their raw condition
    trees: render-time evaluation against the live properties (the
    Blockscope select_variants shape) needs no key enumeration and no
    property-domain inference. The model refs cover every referenced
    model regardless, feeding the spec 5.3 texture fallback.
    """
    out: list[tuple[str, list[dict[str, Any]]]] = []
    variants = state.get("variants")
    if isinstance(variants, dict):
        for key, value in variants.items():
            variant = value[0] if isinstance(value, list) and value else value
            if isinstance(variant, dict) and isinstance(variant.get("model"), str):
                out.append((_normalize_state_key(key), [variant]))
    parts: list[dict[str, Any]] | None = None
    raw = state.get("multipart")
    if isinstance(raw, list) and raw:
        parts = []
        for part in raw:
            if not isinstance(part, dict):
                continue
            when = part.get("when")
            record_when = None
            if isinstance(when, Mapping):
                record_when = json.loads(json.dumps(when))
            apply = part.get("apply")
            entries = apply if isinstance(apply, list) else [apply]
            for entry in entries:
                # Multipart apply arrays are several models rendered
                # together under one condition: flatten to one record per
                # model, all sharing the part's condition.
                if isinstance(entry, dict) and isinstance(entry.get("model"), str):
                    record = {
                        "model": entry["model"],
                        "when": record_when,
                        "x": int(entry.get("x", 0)),
                        "y": int(entry.get("y", 0)),
                    }
                    parts.append(record)
    refs = [v["model"] for _, applies in out for v in applies]
    if parts is not None:
        refs.extend(p["model"] for p in parts)
    return out, parts, refs
