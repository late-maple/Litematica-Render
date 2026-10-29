"""Blockstate JSON parsing: variants and multipart when-merge."""

import itertools
from collections.abc import Mapping
from typing import Any

_KEY_FUSE = 2048



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



def _multipart_applies(
    parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect the apply model refs of every multipart part."""
    applies: list[dict[str, Any]] = []
    for part in parts:
        apply = part.get("apply")
        entries = apply if isinstance(apply, list) else [apply]
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("model"), str):
                applies.append(entry)
    return applies



def _when_matches(
    condition: Mapping[str, Any], values: Mapping[str, str]
) -> bool:
    """Vanilla multipart condition: OR/AND nesting, | alternatives."""
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



def _collect_when(
    condition: Mapping[str, Any], mentioned: dict[str, set[str]]
) -> None:
    """Gather every property and value a condition can distinguish."""
    for key, expected in condition.items():
        if key in ("OR", "AND"):
            if isinstance(expected, list):
                for item in expected:
                    if isinstance(item, Mapping):
                        _collect_when(item, mentioned)
        else:
            mentioned.setdefault(str(key), set()).update(
                str(expected).split("|")
            )



def _multipart_states(
    parts: list[dict[str, Any]], block_id: str
) -> list[tuple[str, list[dict[str, Any]]]] | None:
    """Enumerate when-merged assignments of one multipart blockstate.

    The domain per property is the values its conditions mention plus a
    none bucket for everything else; the projected key therefore carries
    exactly the properties that can change the merged geometry. Redstone
    wire also enumerates power so the per-power dust tint lands in the
    state key. None means the blockstate defers to the builtin table.
    """
    if not parts:
        return None
    mentioned: dict[str, set[str]] = {}
    for part in parts:
        when = part.get("when")
        if isinstance(when, Mapping):
            _collect_when(when, mentioned)
    if not mentioned:
        return None
    domains: dict[str, list[str]] = {}
    if block_id.partition(":")[2] == "redstone_wire":
        # Power never changes geometry but drives the dust tint, so the
        # state key must carry it for the per-power texture variants.
        domains["power"] = [str(power) for power in range(16)]
    for prop, vals in mentioned.items():
        if prop in domains:
            continue
        domain = sorted(vals)
        # The bucket for unmentioned values must equal what real states
        # carry: boolean conditions pair true with false, everything
        # else with none.
        bucket = "false" if vals == {"true"} else "none"
        if bucket not in domain:
            domain.append(bucket)
        domains[prop] = domain
    names = sorted(domains)
    total = 1
    for prop in names:
        total *= len(domains[prop])
    if total > _KEY_FUSE:
        return None
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for combo in itertools.product(*(domains[prop] for prop in names)):
        values = dict(zip(names, combo))
        applies: list[dict[str, Any]] = []
        for part in parts:
            when = part.get("when")
            if when is not None:
                if not isinstance(when, Mapping) or not _when_matches(
                    when, values
                ):
                    continue
            apply = part.get("apply")
            entries = apply if isinstance(apply, list) else [apply]
            for entry in entries:
                if isinstance(entry, dict) and isinstance(
                    entry.get("model"), str
                ):
                    applies.append(entry)
        key = ",".join(f"{prop}={values[prop]}" for prop in names)
        out.append((key, applies))
    return out



def _all_variants(
    state: dict[str, Any], block_id: str = ""
) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[str]]:
    """Group a state file into (state key applies, model refs).

    Variant keys are alternatives and map to one apply. A multipart is
    enumerated over the properties its when-conditions mention (plus
    power for tinted wire), each key holding the applies of every part
    whose condition that assignment satisfies — the vanilla merge. The
    model refs cover every referenced model regardless, feeding the spec
    5.3 texture fallback for deferred states.
    """
    out: list[tuple[str, list[dict[str, Any]]]] = []
    refs: list[str] = []
    variants = state.get("variants")
    if isinstance(variants, dict):
        for key, value in variants.items():
            variant = value[0] if isinstance(value, list) and value else value
            if isinstance(variant, dict) and isinstance(variant.get("model"), str):
                out.append((_normalize_state_key(key), [variant]))
                refs.append(variant["model"])
        return out, refs
    multipart = state.get("multipart")
    if isinstance(multipart, list) and multipart:
        parts = [part for part in multipart if isinstance(part, dict)]
        refs.extend(entry["model"] for entry in _multipart_applies(parts))
        enumerated = _multipart_states(parts, block_id)
        if enumerated is not None:
            out.extend(enumerated)
        elif all(part.get("when") is None for part in parts):
            applies = _multipart_applies(parts)
            if applies:
                out.append(("", applies))
    return out, refs
