"""Quad painter: inverse-basis rasterization with vanilla face lighting."""

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.quads import FACE_LIGHT
from litematic.render.scene.isometric import _Scene

_CUTOUT_ALPHA = 16
_COVER_ALPHA = 250
_FUSE_PIXELS = 1 << 25
# Per-rasterize-call canvas-clipped bbox budget; oversized depth layers
# split into contiguous quad sub-batches under it.
_BATCH_PIXELS = 1 << 20
# A projected quad rasterizes only when its edge vectors are not parallel;
# the threshold is relative (a fraction of |e1|*|e2|) so it holds at every
# tile scale and only true slivers / edge-on planes are dropped.
_PARALLEL_EPS = 1e-4
# Scene quads must stay parallelograms, matching quads.is_parallelogram's
# extract-side eps; pack vertices may legally reach beyond 0..16.
_PARALLELOGRAM_EPS = 1e-3
_LIGHT = np.asarray(FACE_LIGHT, np.float64)


def _paint(
    scene: _Scene, t: int, background: tuple[int, int, int, int]
) -> np.ndarray:
    """Rasterize the scene's quads onto an RGBA uint8 canvas at scale t."""
    try:
        return _paint_inner(scene, t, background)
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError(f"paint: quad rasterization failed: {exc}") from exc


def _paint_inner(
    scene: _Scene, t: int, background: tuple[int, int, int, int]
) -> np.ndarray:
    """Fill the canvas background, then paint depth layers far to near."""
    canvas = np.empty((t * scene.span_h, 2 * t * scene.span_w, 4), np.uint8)
    canvas[:] = background
    if len(scene.x) == 0:
        return canvas
    _assert_parallelograms(scene)
    stacks, group_ids, slot_ids = _texture_stack(scene)
    quad_group = group_ids[scene.texkey_ids]
    quad_slot = slot_ids[scene.texkey_ids]
    sx, sy = _project(scene, t)
    depth = _depth(scene)
    order = np.argsort(depth, kind="stable")
    edges = np.flatnonzero(np.diff(depth[order])) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(order)]))
    for lo, hi in zip(starts.tolist(), stops.tolist()):
        _paint_layer(canvas, scene, order[lo:hi], sx, sy, stacks,
                     quad_group, quad_slot)
    del sx, sy, depth, order
    return canvas


def _assert_parallelograms(scene: _Scene) -> None:
    """Fail fast on non-parallelogram quads, mirroring the extract check."""
    vtx = scene.quad_vtx.astype(np.float64)
    diag = (vtx[:, 0] + vtx[:, 2]) - (vtx[:, 1] + vtx[:, 3])
    if np.any(np.abs(diag) > _PARALLELOGRAM_EPS):
        raise RenderError(
            "paint: non-parallelogram quad reached the painter; "
            "quads must be dropped at extraction time"
        )


def _texture_stack(
    scene: _Scene,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Group scene textures by (h, w) into sorted stacks plus lookup ids."""
    arrays = list(scene.texkeys.values())
    if scene.texkey_ids.size and int(scene.texkey_ids.max()) >= len(arrays):
        raise RenderError("paint: texkey id outside the scene texture table")
    if not arrays:
        return [], np.zeros(0, np.int8), np.zeros(0, np.int64)
    shapes = sorted({arr.shape[:2] for arr in arrays})
    stacks = [
        np.stack([a for a in arrays if a.shape[:2] == shape])
        for shape in shapes
    ]
    group_ids = np.zeros(len(arrays), np.int8)
    slot_ids = np.zeros(len(arrays), np.int64)
    for gid, shape in enumerate(shapes):
        for slot, index in enumerate(
            i for i, a in enumerate(arrays) if a.shape[:2] == shape
        ):
            group_ids[index] = gid
            slot_ids[index] = slot
    return stacks, group_ids, slot_ids


def _project(scene: _Scene, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Project quad vertices to anchored screen columns and rows."""
    fx = scene.quad_vtx[..., 0].astype(np.float64) / 16.0
    fy = scene.quad_vtx[..., 1].astype(np.float64) / 16.0
    fz = scene.quad_vtx[..., 2].astype(np.float64) / 16.0
    bx = scene.x[:, None].astype(np.float64)
    by = scene.y[:, None].astype(np.float64)
    bz = scene.z[:, None].astype(np.float64)
    col = 2.0 * t * (bx + fx - bz - fz) + 2.0 * t * (int(scene.z.max()) + 1)
    row = t * (bx + bz + fx + fz) - 2.0 * t * (by + fy)
    row += 2.0 * t * (int(scene.y.max()) + 1)
    return col, row


def _depth(scene: _Scene) -> np.ndarray:
    """Painter depth: the block sum plus the centroid in block fractions."""
    centroid = scene.quad_vtx.astype(np.float64).mean(axis=1) / 16.0
    return (
        scene.x.astype(np.float64)
        + scene.y.astype(np.float64)
        + scene.z.astype(np.float64)
        + centroid.sum(axis=1)
    )


def _paint_layer(
    canvas: np.ndarray,
    scene: _Scene,
    layer: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    stacks: list[np.ndarray],
    quad_group: np.ndarray,
    quad_slot: np.ndarray,
) -> None:
    """Rasterize one depth layer group by group: covers first, then blends."""
    layer_groups = quad_group[layer]
    for gid, stack in enumerate(stacks):
        sub = layer[layer_groups == gid]
        if len(sub) == 0:
            continue
        _paint_group(
            canvas, scene, sub, sx, sy, stack, quad_slot, stack.shape[1:3]
        )


def _paint_group(
    canvas: np.ndarray,
    scene: _Scene,
    layer: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    textures: np.ndarray,
    quad_slot: np.ndarray,
    tex_hw: tuple[int, int],
) -> None:
    """Rasterize one depth layer restricted to a single texture shape."""
    quad_sx = sx[layer]
    quad_sy = sy[layer]
    uv = scene.quad_uv[layer].astype(np.float64)
    ao = scene.quad_ao[layer].astype(np.float64)
    shade = scene.quad_shade[layer]
    tex = textures[quad_slot[layer]]
    dbl = np.nonzero(scene.quad_dbl[layer])[0]
    if len(dbl):
        # Double-sided quads paint both windings in the same layer.
        quad_sx = np.concatenate([quad_sx, quad_sx[dbl, ::-1]])
        quad_sy = np.concatenate([quad_sy, quad_sy[dbl, ::-1]])
        uv = np.concatenate([uv, uv[dbl, ::-1]])
        ao = np.concatenate([ao, ao[dbl, ::-1]])
        shade = np.concatenate([shade, shade[dbl]])
        tex = np.concatenate([tex, tex[dbl]])
    height, width = canvas.shape[:2]
    pixels = _bbox_pixels(quad_sx, quad_sy, height, width)[0]
    chunks = _layer_chunks(pixels, _BATCH_PIXELS)
    deferred = []
    for lo, hi in chunks:
        rows, cols, values = _rasterize(
            quad_sx[lo:hi],
            quad_sy[lo:hi],
            uv[lo:hi],
            ao[lo:hi],
            shade[lo:hi],
            tex[lo:hi],
            tex_hw,
            height,
            width,
        )
        alpha = values[:, 3]
        cover = alpha >= _COVER_ALPHA
        canvas[rows[cover], cols[cover]] = values[cover]
        blend = (alpha >= _CUTOUT_ALPHA) & ~cover
        if blend.any():
            deferred.append((rows[blend], cols[blend], values[blend]))
    for rows, cols, values in deferred:
        _blend_pixels(canvas, rows, cols, values)


def _bbox_pixels(
    sx: np.ndarray, sy: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-quad canvas-clipped bbox: (pixels, width, height, lo_col, lo_row)."""
    lo_c = np.maximum(np.floor(sx.min(axis=1)), 0.0)
    hi_c = np.minimum(np.ceil(sx.max(axis=1)), float(width))
    lo_r = np.maximum(np.floor(sy.min(axis=1)), 0.0)
    hi_r = np.minimum(np.ceil(sy.max(axis=1)), float(height))
    width_q = np.maximum((hi_c - lo_c).astype(np.int64), 0)
    height_q = np.maximum((hi_r - lo_r).astype(np.int64), 0)
    return width_q * height_q, width_q, height_q, lo_c, lo_r


def _layer_chunks(pixels: np.ndarray, budget: int) -> list[tuple[int, int]]:
    """Contiguous [start, stop) quad ranges whose bbox sum fits budget."""
    total = int(pixels.sum())
    if total <= budget:
        return [(0, len(pixels))]
    cum = np.cumsum(pixels)
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < len(pixels):
        consumed = int(cum[start - 1]) if start else 0
        stop = int(np.searchsorted(cum, consumed + budget, side="right"))
        if stop <= start:
            stop = start + 1  # a single quad over budget paints alone
        chunks.append((start, stop))
        start = stop
    return chunks


def _rasterize(
    sx: np.ndarray,
    sy: np.ndarray,
    uv: np.ndarray,
    ao: np.ndarray,
    shade: np.ndarray,
    tex: np.ndarray,
    tex_hw: tuple[int, int],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample each quad's canvas-clipped bbox through the inverse basis."""
    counts, width_q, height_q, lo_c, lo_r = _bbox_pixels(
        sx, sy, height, width
    )
    total = int(counts.sum())
    if total == 0:
        empty = np.zeros(0, np.int64)
        return empty, empty, np.zeros((0, 4), np.uint8)
    quad_of = np.repeat(np.arange(len(sx), dtype=np.int64), counts)
    starts = np.cumsum(counts) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    span = np.maximum(width_q[quad_of], 1)
    cols = lo_c[quad_of].astype(np.int64) + within % span
    rows = lo_r[quad_of].astype(np.int64) + within // span
    u, v, inside = _inverse_basis(
        sx, sy, quad_of, cols + 0.5, rows + 0.5
    )
    if not inside.any():
        empty = np.zeros(0, np.int64)
        return empty, empty, np.zeros((0, 4), np.uint8)
    return _shade_pixels(
        quad_of[inside],
        u[inside],
        v[inside],
        rows[inside],
        cols[inside],
        uv,
        ao,
        shade,
        tex,
        tex_hw,
    )


def _inverse_basis(
    sx: np.ndarray,
    sy: np.ndarray,
    quad_of: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve p = p0 + u*(p1-p0) + v*(p3-p0); flag non-degenerate hits."""
    p0x = sx[quad_of, 0]
    p0y = sy[quad_of, 0]
    e1x = sx[quad_of, 1] - p0x
    e1y = sy[quad_of, 1] - p0y
    e2x = sx[quad_of, 3] - p0x
    e2y = sy[quad_of, 3] - p0y
    det = e1x * e2y - e2x * e1y
    dx = px - p0x
    dy = py - p0y
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u = (dx * e2y - dy * e2x) / det
        v = (e1x * dy - e1y * dx) / det
    norm = np.hypot(e1x, e1y) * np.hypot(e2x, e2y)
    inside = (
        (np.abs(det) > _PARALLEL_EPS * norm)
        & (u >= 0.0)
        & (u < 1.0)
        & (v >= 0.0)
        & (v < 1.0)
    )
    return u, v, inside


def _shade_pixels(
    quad: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    uv: np.ndarray,
    ao: np.ndarray,
    shade: np.ndarray,
    tex: np.ndarray,
    tex_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather texels, apply face light times bilinear AO, and return RGBA."""
    du = u * (uv[quad, 1, 0] - uv[quad, 0, 0]) + v * (
        uv[quad, 3, 0] - uv[quad, 0, 0]
    )
    dv = u * (uv[quad, 1, 1] - uv[quad, 0, 1]) + v * (
        uv[quad, 3, 1] - uv[quad, 0, 1]
    )
    # Scale 0..16 uv space to the texture's native size; clip, no wrap.
    tex_h, tex_w = tex_hw
    tx = np.clip(
        np.floor((uv[quad, 0, 0] + du) * (tex_w / 16.0)), 0, tex_w - 1
    ).astype(np.int64)
    ty = np.clip(
        np.floor((uv[quad, 0, 1] + dv) * (tex_h / 16.0)), 0, tex_h - 1
    ).astype(np.int64)
    brightness = _LIGHT[shade[quad]] * (
        ao[quad, 0]
        + u * (ao[quad, 1] - ao[quad, 0])
        + v * (ao[quad, 3] - ao[quad, 0])
    )
    pixels = tex[quad, ty, tx].astype(np.float64)
    lit = np.rint(pixels[:, :3] * brightness[:, None])
    pixels[:, :3] = np.clip(lit, 0.0, 255.0)
    return rows, cols, pixels.astype(np.uint8)


def _blend_pixels(
    canvas: np.ndarray, rows: np.ndarray, cols: np.ndarray, src: np.ndarray
) -> None:
    """Over-composite source pixels onto the canvas at the given cells."""
    dst = canvas[rows, cols].astype(np.uint16)
    src_u = src.astype(np.uint16)
    alpha = src_u[:, 3:4]
    out_rgb = np.rint((src_u[:, :3] * alpha + dst[:, :3] * (255 - alpha)) / 255)
    out_alpha = np.rint(src_u[:, 3:4] + dst[:, 3:4] * (255 - alpha) / 255)
    blended = np.concatenate(
        [np.clip(out_rgb, 0, 255), np.clip(out_alpha, 0, 255)], axis=1
    ).astype(np.uint8)
    canvas[rows, cols] = blended


def _choose_scale(
    span_w: int, span_h: int, max_size: int, supersample: int
) -> tuple[int, int]:
    """Pick tile scale and supersample factor within the canvas pixel fuse."""
    for ss in range(supersample, 0, -1):
        t = max(1, min(max_size * ss // (2 * span_w), max_size * ss // span_h))
        width = 2 * t * span_w
        height = t * span_h
        fits = width <= max_size * ss and height <= max_size * ss
        if (fits or ss == 1) and width * height <= _FUSE_PIXELS:
            return t, ss
    raise RenderError(
        f"scale: spans {span_w}x{span_h} exceed canvas budget at "
        f"max_size={max_size}"
    )


def _downsample(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Box-filter downsample RGBA uint8 with alpha-weighted color."""
    in_h, in_w = img.shape[:2]
    if out_w == in_w and out_h == in_h:
        return img.copy()
    rows = np.minimum(np.arange(in_h) * out_h // in_h, out_h - 1)
    cols = np.minimum(np.arange(in_w) * out_w // in_w, out_w - 1)
    flat = (rows[:, None] * out_w + cols[None, :]).ravel()
    size = out_h * out_w
    alpha = img[..., 3].ravel().astype(np.float64) / 255.0
    weight = np.bincount(flat, weights=alpha, minlength=size)
    count = np.bincount(flat, minlength=size).astype(np.float64)
    out_a = np.rint(weight / count * 255.0)
    covered = (weight > 0) & (out_a > 0)
    out = np.zeros((size, 4), np.uint8)
    out[:, 3] = np.clip(out_a, 0, 255).astype(np.uint8)
    for channel in range(3):
        values = img[..., channel].ravel().astype(np.float64)
        summed = np.bincount(flat, weights=values * alpha, minlength=size)
        mean = np.zeros(size)
        mean[covered] = summed[covered] / weight[covered]
        out[:, channel] = np.clip(np.rint(mean), 0, 255).astype(np.uint8)
    return out.reshape(out_h, out_w, 4)
