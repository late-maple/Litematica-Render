"""Quad painter: z-buffered inverse-basis rasterization, camera-locked light."""

import numpy as np

from litematic.render.errors import RenderError
from litematic.render.quads import TIER_NORMALS
from litematic.render.scene.isometric import _Scene
from litematic.render.scene.view import COL_SCALE, ROW_SCALE

_CUTOUT_ALPHA = 16
_COVER_ALPHA = 250
_FUSE_PIXELS = 1 << 25
# Per-rasterize-call canvas-clipped bbox budget; oversized quad batches
# split into contiguous sub-batches under it.
_BATCH_PIXELS = 1 << 20
# A projected quad rasterizes only when its edge vectors are not parallel;
# the threshold is relative (a fraction of |e1|*|e2|) so it holds at every
# tile scale and only true slivers / edge-on planes are dropped.
_PARALLEL_EPS = 1e-4
# Scene quads must stay parallelograms, matching quads.is_parallelogram's
# extract-side eps; pack vertices may legally reach beyond 0..16.
_PARALLELOGRAM_EPS = 1e-3
# Camera-locked light in view space, solved so the preset view keeps the
# vanilla face lights (up 1.0, south 0.8, east 0.6).
_LIGHT_VIEW = np.array([-(2**0.5) / 10, (6**0.5) / 10, 0.8 * 3**0.5])


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
    """Fill the background, then z-buffer quads far to near."""
    canvas = np.empty((t * scene.span_h, t * scene.span_w, 4), np.uint8)
    canvas[:] = background
    if len(scene.x) == 0:
        return canvas
    if scene.axes is None:
        raise RenderError("paint: scene is missing its camera axes")
    _assert_parallelograms(scene)
    stacks, group_ids, slot_ids = _texture_stack(scene)
    quad_group = group_ids[scene.texkey_ids]
    quad_slot = slot_ids[scene.texkey_ids]
    light = _face_light(scene)
    sx, sy, sd = _project(scene, t)
    # Far-to-near paint order: the scatter's last write wins, so the
    # nearest fragment of any overlapping batch lands on the canvas.
    # Translucent fragments flush per chunk: chunks are already in
    # far-to-near order, so blending is globally ordered while the
    # pending fragment memory stays bounded by the chunk budget.
    order = np.argsort(-sd.mean(axis=1), kind="stable")
    height, width = canvas.shape[:2]
    zbuf = np.full(canvas.shape[:2], np.inf, np.float32)
    pixels = _bbox_pixels(sx, sy, height, width)[0][order]
    for lo, hi in _layer_chunks(pixels, _BATCH_PIXELS):
        idx = order[lo:hi]
        chunk_blends: list = []
        for gid, stack in enumerate(stacks):
            sub = idx[quad_group[idx] == gid]
            if len(sub) == 0:
                continue
            _paint_group(
                canvas, zbuf, chunk_blends, scene, sub, sx, sy, sd, light,
                stack, quad_slot, stack.shape[1:3],
            )
        _blend_translucent(canvas, zbuf, chunk_blends)
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


def _project(
    scene: _Scene, t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project quad corners to screen cols, rows, and camera depths."""
    axes = scene.axes
    if axes is None:
        raise RenderError("paint: scene is missing its camera axes")
    fx = scene.quad_vtx[..., 0].astype(np.float64) / 16.0
    fy = scene.quad_vtx[..., 1].astype(np.float64) / 16.0
    fz = scene.quad_vtx[..., 2].astype(np.float64) / 16.0
    wx = scene.x[:, None].astype(np.float64) + fx
    wy = scene.y[:, None].astype(np.float64) + fy
    wz = scene.z[:, None].astype(np.float64) + fz
    del fx, fy, fz
    col = wx * axes.right[0] + wy * axes.right[1] + wz * axes.right[2]
    row = -(wx * axes.up[0] + wy * axes.up[1] + wz * axes.up[2])
    sx = COL_SCALE * t * col
    sy = ROW_SCALE * t * row
    del col, row
    sx -= sx.min()
    sy -= sy.min()
    sd = -(
        wx * axes.toward[0] + wy * axes.toward[1] + wz * axes.toward[2]
    )
    del wx, wy, wz
    return sx, sy, sd.astype(np.float32)


def _face_light(scene: _Scene) -> np.ndarray:
    """Per-quad signed light from the camera-locked directional light.

    Unclipped: double-sided quads negate it for their back winding, and
    the rasterizer clamps it to [0, 1] per fragment.
    """
    axes = scene.axes
    if axes is None:
        raise RenderError("paint: scene is missing its camera axes")
    l_world = (
        axes.right * _LIGHT_VIEW[0]
        + axes.up * _LIGHT_VIEW[1]
        + axes.toward * _LIGHT_VIEW[2]
    )
    return TIER_NORMALS[scene.quad_shade] @ l_world


def _blend_translucent(
    canvas: np.ndarray, zbuf: np.ndarray, blends: list
) -> None:
    """Blend deferred translucent fragments back to front over the depth."""
    if not blends:
        return
    rows = np.concatenate([part[0] for part in blends])
    cols = np.concatenate([part[1] for part in blends])
    values = np.concatenate([part[2] for part in blends])
    depth = np.concatenate([part[3] for part in blends])
    keep = depth <= zbuf[rows, cols]
    rows, cols, values, depth = (
        rows[keep], cols[keep], values[keep], depth[keep],
    )
    if not len(rows):
        return
    width = canvas.shape[1]
    flat = rows.astype(np.int64) * width + cols
    # Per pixel back to front: far-to-near first, then grouped by pixel;
    # the stable passes keep array order for exact (pixel, depth) ties.
    order = np.argsort(-depth, kind="stable")
    order = order[np.argsort(flat[order], kind="stable")]
    flat = flat[order]
    depth = depth[order]
    values = values[order]
    # Coincident fragments collapse to the last one, matching the
    # single-write-per-depth rule of the opaque path.
    dup = np.zeros(len(flat), np.bool_)
    np.logical_and(flat[:-1] == flat[1:], depth[:-1] == depth[1:], out=dup[:-1])
    sel = np.flatnonzero(~dup)
    flat = flat[sel]
    values = values[sel]
    # Blend one stack layer at a time: every pixel contributes at most one
    # fragment per layer, so the Python loop count is the stack depth and
    # each iteration is a vectorized batch over distinct pixels.
    boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    lengths = np.diff(np.append(starts, len(flat)))
    layer = np.arange(len(flat)) - np.repeat(starts, lengths)
    blend_rows = flat // width
    blend_cols = flat - blend_rows * width
    for j in range(int(lengths.max())):
        pick = layer == j
        _blend_pixels(canvas, blend_rows[pick], blend_cols[pick], values[pick])


def _paint_group(
    canvas: np.ndarray,
    zbuf: np.ndarray,
    blends: list,
    scene: _Scene,
    quads: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sd: np.ndarray,
    light: np.ndarray,
    textures: np.ndarray,
    quad_slot: np.ndarray,
    tex_hw: tuple[int, int],
) -> None:
    """Z-buffer rasterize one quad batch restricted to a texture shape."""
    quad_sx = sx[quads]
    quad_sy = sy[quads]
    quad_sd = sd[quads]
    uv = scene.quad_uv[quads].astype(np.float64)
    ao = scene.quad_ao[quads].astype(np.float64)
    quad_light = light[quads]
    tex = textures[quad_slot[quads]]
    dbl = np.nonzero(scene.quad_dbl[quads])[0]
    # Double-sided decals face both ways and both windings land on the
    # same pixels, so they take the light magnitude: the visible side
    # keeps its brightness from every camera direction.
    quad_light[dbl] = np.abs(quad_light[dbl])
    if len(dbl):
        quad_sx = np.concatenate([quad_sx, quad_sx[dbl, ::-1]])
        quad_sy = np.concatenate([quad_sy, quad_sy[dbl, ::-1]])
        quad_sd = np.concatenate([quad_sd, quad_sd[dbl, ::-1]])
        uv = np.concatenate([uv, uv[dbl, ::-1]])
        ao = np.concatenate([ao, ao[dbl, ::-1]])
        quad_light = np.concatenate([quad_light, quad_light[dbl]])
        tex = np.concatenate([tex, tex[dbl]])
    height, width = canvas.shape[:2]
    pixels = _bbox_pixels(quad_sx, quad_sy, height, width)[0]
    for lo, hi in _layer_chunks(pixels, _BATCH_PIXELS):
        rows, cols, values, depth = _rasterize(
            quad_sx[lo:hi],
            quad_sy[lo:hi],
            quad_sd[lo:hi],
            uv[lo:hi],
            ao[lo:hi],
            quad_light[lo:hi],
            tex[lo:hi],
            tex_hw,
            height,
            width,
        )
        if len(rows) == 0:
            continue
        alpha = values[:, 3]
        visible = depth <= zbuf[rows, cols]
        cover = visible & (alpha >= _COVER_ALPHA)
        if cover.any():
            _write_cover(
                canvas, zbuf, rows[cover], cols[cover], values[cover],
                depth[cover], width,
            )
        blend = visible & (alpha >= _CUTOUT_ALPHA) & (alpha < _COVER_ALPHA)
        if blend.any():
            blends.append(
                (rows[blend], cols[blend], values[blend], depth[blend])
            )


def _write_cover(
    canvas: np.ndarray,
    zbuf: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    depth: np.ndarray,
    width: int,
) -> None:
    """Write one batch's cover fragments, nearest depth winning per pixel.

    Every fragment of a batch tests against the pre-batch zbuffer, so
    fragments overlapping inside one batch must be resolved here: the
    batch's nearest fragment takes the pixel and array order breaks
    exact-depth ties. The composite key sorts pixels together with the
    nearest depth last, so the final fragment of every pixel run wins.
    """
    flat = rows.astype(np.int64) * width + cols
    dep32 = np.ascontiguousarray(depth, dtype=np.float32)
    bits = dep32.view(np.uint32)
    # f32 total order with the nearest depth mapping largest, so that a
    # stable ascending sort leaves the winner at the end of each run.
    ordered = np.where(
        bits >= np.uint32(0x80000000), bits, np.uint32(0x7FFFFFFF) - bits
    )
    key = (flat << 32) + ordered.astype(np.int64)
    order = np.argsort(key, kind="stable")
    sflat = key[order] >> 32
    last = np.empty(len(order), np.bool_)
    last[-1] = True
    np.not_equal(sflat[:-1], sflat[1:], out=last[:-1])
    winners = order[last]
    wf = sflat[last]
    wr = wf // width
    zbuf[wr, wf - wr * width] = dep32[winners]
    canvas[wr, wf - wr * width] = values[winners]


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
    sd: np.ndarray,
    uv: np.ndarray,
    ao: np.ndarray,
    light: np.ndarray,
    tex: np.ndarray,
    tex_hw: tuple[int, int],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample each quad's canvas-clipped bbox through the inverse basis."""
    counts, width_q, height_q, lo_c, lo_r = _bbox_pixels(
        sx, sy, height, width
    )
    total = int(counts.sum())
    if total == 0:
        empty = np.zeros(0, np.int64)
        return empty, empty, np.zeros((0, 4), np.uint8), empty
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
        return empty, empty, np.zeros((0, 4), np.uint8), empty
    quad = quad_of[inside]
    u = u[inside]
    v = v[inside]
    # Ortho depth is linear over the parallelogram: corners ride the same
    # (u, v) affine combo as the UVs, so interpolation is exact.
    depth = sd[quad, 0] + u * (sd[quad, 1] - sd[quad, 0]) + v * (
        sd[quad, 3] - sd[quad, 0]
    )
    return _shade_pixels(
        quad,
        u,
        v,
        rows[inside],
        cols[inside],
        depth,
        uv,
        ao,
        light,
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
    depth: np.ndarray,
    uv: np.ndarray,
    ao: np.ndarray,
    light: np.ndarray,
    tex: np.ndarray,
    tex_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    brightness = np.clip(light[quad], 0.0, 1.0) * (
        ao[quad, 0]
        + u * (ao[quad, 1] - ao[quad, 0])
        + v * (ao[quad, 3] - ao[quad, 0])
    )
    pixels = tex[quad, ty, tx].astype(np.float64)
    lit = np.rint(pixels[:, :3] * brightness[:, None])
    pixels[:, :3] = np.clip(lit, 0.0, 255.0)
    return rows, cols, pixels.astype(np.uint8), depth


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
        t = max(1, min(max_size * ss // span_w, max_size * ss // span_h))
        width = t * span_w
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
