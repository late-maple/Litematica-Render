"""Model element geometry: boxes compiled to textured quads."""

import math
from typing import Any, Callable, cast

import numpy as np

from litematic.render.pack.texture import _rotate
from litematic.render.quads import (
    QUAD_DTYPE,
    TIER_DOWN,
    TIER_EAST,
    TIER_NORTH,
    TIER_SOUTH,
    TIER_UP,
    TIER_WEST,
    is_parallelogram,
)

_OPAQUE_ALPHA = 250.0



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



_VEC = {
    "up": (0, 1, 0),
    "down": (0, -1, 0),
    "north": (0, 0, -1),
    "south": (0, 0, 1),
    "west": (-1, 0, 0),
    "east": (1, 0, 0),
}



_ROT_AXIS = {"x": 0, "y": 1, "z": 2}



_RESCALE_FACTORS = {22.5: 1.0823922, 45.0: 1.4142135}



_FACE_GEOM: dict[int, tuple[tuple[int, int, int], ...]] = {
    TIER_DOWN: ((0, 0, 16), (16, 0, 16), (16, 0, 0), (0, 0, 0)),
    TIER_UP: ((0, 16, 0), (16, 16, 0), (16, 16, 16), (0, 16, 16)),
    TIER_NORTH: ((16, 16, 0), (0, 16, 0), (0, 0, 0), (16, 0, 0)),
    TIER_SOUTH: ((0, 16, 16), (16, 16, 16), (16, 0, 16), (0, 0, 16)),
    TIER_WEST: ((0, 16, 16), (0, 16, 0), (0, 0, 0), (0, 0, 16)),
    TIER_EAST: ((16, 16, 0), (16, 16, 16), (16, 0, 16), (16, 0, 0)),
}
# Vanilla element rotation: 90-degree-free angles about one axis through an
# origin, optionally rescaled on the perpendicular axes by 1/cos(|angle|).



def _corner_tuple(point: Any) -> tuple[float, float, float]:
    """Convert one vertex row to a plain three-float tuple."""
    x, y, z = (float(value) for value in point)
    return (x, y, z)



def _sorted_corners(quad: Any) -> list[tuple[float, float, float]]:
    """Return a quad's corners as a sorted, order-independent list."""
    return sorted(_corner_tuple(point) for point in quad["vtx"])


_UNIT_CORNERS: dict[int, list[tuple[float, float, float]]] = {
    tier: sorted(_corner_tuple(corner) for corner in corners)
    for tier, corners in _FACE_GEOM.items()
}



# Unit-cube corners per tier as sorted order-independent lists: blockstate
# rotations map a face onto its own square with reversed winding, so an
# element-wise comparison would wrongly reject rotated opaque cubes.




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
    """Compile model elements into world-space quads for one variant."""
    rows: list[tuple[Any, Any, str, int, bool]] = []
    kx = (int(x_rot) // 90) % 4
    ky = (int(y_rot) // 90) % 4
    for element in elements:
        if not isinstance(element, dict):
            continue
        bounds = _element_bounds(element)
        faces = element.get("faces")
        if bounds is None or not isinstance(faces, dict):
            continue
        try:
            rotation = _element_rotation(element)
        except ValueError:
            continue
        thin = _thin_axes(bounds)
        if len(thin) > 1:
            continue
        rotated = rotation is not None
        # Zero-thickness planes and rotated elements have non-axis-aligned
        # or paired back faces: spec marks them dbl (double-sided, no AO).
        double: bool = bool(thin) or rotated
        shaded = bool(element.get("shade", True))
        for face_name, tier in _FACE_TIER.items():
            face = faces.get(face_name)
            if not isinstance(face, dict):
                continue
            if thin and _VEC[face_name][thin[0]] == 0:
                continue
            texkey = texkey_for(face, textures, ns)
            if texkey is None:
                continue
            corners = _box_corners(bounds, tier)
            uvs = _face_uvs(face, tier, corners)
            if uvs is None:
                continue
            if double or not shaded:
                shade: int = TIER_UP
            else:
                shade = _FACE_TIER[_rotate(face_name, x_rot, y_rot)]
            if rotation is not None:
                axis, angle, origin, factor = rotation
                corners = [
                    _rotate_element_point(corner, axis, angle, origin, factor)
                    for corner in corners
                ]
            rows.append(
                (_rotate_corners(corners, kx, ky), uvs, texkey, shade, double)
            )
    if not rows:
        return None
    quads = np.empty(len(rows), dtype=QUAD_DTYPE)
    for index, (corners, uvs, texkey, shade, dbl) in enumerate(rows):
        quads[index] = (corners, uvs, texkey, shade, dbl)
    keep = is_parallelogram(quads)
    if not keep.any():
        return None
    return quads[keep]



def _thin_axes(
    bounds: tuple[float, float, float, float, float, float],
) -> list[int]:
    """Indices of the axes where from equals to (zero-thickness planes)."""
    x0, y0, z0, x1, y1, z1 = bounds
    sizes = (x1 - x0, y1 - y0, z1 - z0)
    return [axis for axis, size in enumerate(sizes) if size == 0.0]



def _element_rotation(
    element: dict[str, Any],
) -> tuple[int, float, tuple[float, float, float], float] | None:
    """Parse rotation into (axis, angle, origin, rescale factor) or None."""
    rot = element.get("rotation")
    if rot is None:
        return None
    if not isinstance(rot, dict):
        raise ValueError("element rotation must be a mapping")
    axis = _ROT_AXIS.get(rot.get("axis", "y"))
    if axis is None:
        raise ValueError("element rotation has an unknown axis")
    angle = rot.get("angle", 0.0)
    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        raise ValueError("element rotation angle must be a number")
    angle = float(angle)
    if not -45.0 <= angle <= 45.0 or angle % 22.5 != 0.0:
        raise ValueError("element rotation angle out of the vanilla set")
    origin = rot.get("origin", (8.0, 8.0, 8.0))
    if not isinstance(origin, (list, tuple)) or len(origin) != 3:
        raise ValueError("element rotation origin must be three numbers")
    try:
        point = tuple(float(value) for value in origin)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("element rotation origin must be three numbers") from exc
    if not all(math.isfinite(value) for value in point):
        raise ValueError("element rotation origin must be finite")
    rescale = bool(rot.get("rescale", False))
    factor = _RESCALE_FACTORS.get(abs(angle), 1.0) if rescale else 1.0
    return (axis, angle, cast("tuple[float, float, float]", point), factor)



def _rotate_element_point(
    point: tuple[float, float, float],
    axis: int,
    angle: float,
    origin: tuple[float, float, float],
    factor: float,
) -> tuple[float, float, float]:
    """Rotate one point by angle about the axis through origin, rescaled."""
    rel = [point[i] - origin[i] for i in range(3)]
    rad = math.radians(angle)
    cos, sin = math.cos(rad), math.sin(rad)
    if axis == 0:
        rel[1], rel[2] = rel[1] * cos - rel[2] * sin, rel[1] * sin + rel[2] * cos
    elif axis == 1:
        rel[0], rel[2] = rel[0] * cos + rel[2] * sin, -rel[0] * sin + rel[2] * cos
    else:
        rel[0], rel[1] = rel[0] * cos - rel[1] * sin, rel[0] * sin + rel[1] * cos
    if factor != 1.0:
        for i in range(3):
            if i != axis:
                rel[i] *= factor
    return (rel[0] + origin[0], rel[1] + origin[1], rel[2] + origin[2])



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
        uvs = [_auto_uv(tier, x, y, z) for (x, y, z) in corners]
    else:
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
        uvs = [(u1, v1), (u2, v1), (u2, v2), (u1, v2)]
    rotation = face.get("rotation")
    if rotation is None:
        return uvs
    if isinstance(rotation, bool) or not isinstance(rotation, (int, float)):
        return None
    if rotation not in (0, 90, 180, 270):
        return None
    shift = int(rotation) // 90
    if shift:
        # The corner ring runs clockwise on the face, so vanilla's clockwise
        # texture rotation takes each corner's coords from shift slots back.
        uvs = uvs[-shift:] + uvs[:-shift]
    return uvs



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
