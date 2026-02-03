#!/usr/bin/env python3
"""
src/support_tube.py

Canonical support-tube geometry builder for PlotEmbeddings.

Canonical mask method
---------------------
We use ONE method for the paper/pipeline:

  trimmed_convex_hull:
    - per u-slice, trim outliers by radius around a robust center (median/mean)
    - compute convex hull of the retained points
    - rasterize hull to a binary mask on a global XY grid

Phase A / Phase B
-----------------
Phase A:
  - build per-slice binary masks (U, Y, X) using trimmed convex hull

Phase B:
  - optionally fill gaps along U (copy-nearest strategy)
  - optionally smooth along U (float volume in [0,1])

Geometry:
  - compute_support_tube_mesh() -> vertices + faces for Mesh3d
  - compute_support_tube_rings() -> polylines (Scatter3d) from mask contours

Conventions
-----------
Grid arrays:
- XY mask is (Y, X) with y as rows, x as cols (matches np.histogram2d layout).
- Volume is (U, Y, X) where U indexes log10(time) bins.

Mapping:
- u = log10(max(time, time_eps))
- for plotting: z = 10**u (suitable for a log z-axis)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes, find_contours
from matplotlib.path import Path


# ---------------------------- small utilities -----------------------------


def _interp_centers(idx: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """idx: float array in [0, n-1], centers: (n,) array."""
    n = len(centers)
    if n == 0:
        return np.array([], dtype=float)
    grid = np.arange(n, dtype=float)
    return np.interp(idx, grid, centers.astype(float))


def _largest_contour(contours: List[np.ndarray]) -> Optional[np.ndarray]:
    if not contours:
        return None
    lengths = [c.shape[0] for c in contours]
    return contours[int(np.argmax(lengths))]


def _make_disk(radius: int) -> np.ndarray:
    """Binary structuring element disk in 2D."""
    r = int(radius)
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= (r * r)


def _make_ball(radius: int) -> np.ndarray:
    """Binary structuring element ball in 3D (U,Y,X)."""
    r = int(radius)
    if r <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy + zz * zz) <= (r * r)


def _fill_u_gaps_copy_nearest(masks: np.ndarray) -> np.ndarray:
    """
    Fill gaps along the u-axis so the support is temporally connected.

    Strategy:
      - Leading/trailing empty region: copy nearest non-empty slice outward.
      - Internal gaps: copy nearest endpoint into the gap (ties split at midpoint).
    """
    m = masks.copy()
    U = m.shape[0]
    has = m.reshape(U, -1).any(axis=1)

    idx = np.where(has)[0]
    if idx.size == 0:
        return m

    first = int(idx[0])
    last = int(idx[-1])

    # leading / trailing
    for u in range(0, first):
        m[u] |= m[first]
    for u in range(last + 1, U):
        m[u] |= m[last]

    # internal gaps
    for a, b in zip(idx[:-1], idx[1:]):
        a = int(a)
        b = int(b)
        if b <= a + 1:
            continue
        mid = (a + b) // 2
        for u in range(a + 1, b):
            m[u] |= m[a] if (u <= mid) else m[b]

    return m


# --------------------- canonical per-slice mask builder --------------------


def _mask_xy_for_slice(
    xsamp: np.ndarray,
    ysamp: np.ndarray,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    mass_frac: float,
    trim_center: str,
    min_points_per_slice: int,
    reason_counts: Dict[str, int],
) -> np.ndarray:
    """
    Canonical per-slice mask: trimmed convex hull rasterized on the XY grid.

    Steps:
      1) compute robust center (median/mean)
      2) keep closest mass_frac points by radius
      3) compute convex hull on retained points
      4) rasterize polygon onto grid cell centers

    Returns:
      mask (Y, X) boolean array

    Side effects:
      updates reason_counts[...] for invalid slice causes
    """
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)

    grid_res_x = x_edges.size - 1
    grid_res_y = y_edges.size - 1
    if grid_res_x <= 0 or grid_res_y <= 0 or grid_res_x != grid_res_y:
        raise ValueError("x_edges and y_edges must be length grid_res_xy+1 and match each other.")

    # empty by default
    mask = np.zeros((grid_res_y, grid_res_x), dtype=bool)

    xsamp = np.asarray(xsamp, dtype=float)
    ysamp = np.asarray(ysamp, dtype=float)
    m = np.isfinite(xsamp) & np.isfinite(ysamp)
    xs = xsamp[m]
    ys = ysamp[m]
    n = int(xs.size)

    if n < int(min_points_per_slice):
        reason_counts["too_few_points"] = reason_counts.get("too_few_points", 0) + 1
        return mask

    mass_frac = float(mass_frac)
    if not (0.0 < mass_frac <= 1.0):
        raise ValueError(f"mass_frac must be in (0,1], got {mass_frac}")

    trim_center = str(trim_center).lower().strip()
    if trim_center not in {"median", "mean"}:
        raise ValueError("trim_center must be 'median' or 'mean'")

    # center
    if trim_center == "mean":
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
    else:
        cx, cy = float(np.median(xs)), float(np.median(ys))

    # radial trim
    dx = xs - cx
    dy = ys - cy
    r2 = dx * dx + dy * dy

    k = int(np.ceil(mass_frac * n))
    k = max(3, min(k, n))

    order = np.argsort(r2)
    keep_idx = order[:k]
    xr = xs[keep_idx]
    yr = ys[keep_idx]

    if xr.size < 3:
        reason_counts["too_few_after_trim"] = reason_counts.get("too_few_after_trim", 0) + 1
        return mask

    pts = np.column_stack([xr, yr])

    # convex hull
    try:
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]  # ordered
    except Exception:
        reason_counts["hull_failed"] = reason_counts.get("hull_failed", 0) + 1
        return mask

    if verts.shape[0] < 3:
        reason_counts["degenerate_hull"] = reason_counts.get("degenerate_hull", 0) + 1
        return mask

    # rasterize to grid cell centers
    xs_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    ys_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    XX, YY = np.meshgrid(xs_centers, ys_centers)  # (Y,X)
    centers = np.column_stack([XX.ravel(), YY.ravel()])

    poly = Path(verts, closed=True)
    inside = poly.contains_points(centers)
    mask = inside.reshape((grid_res_y, grid_res_x))

    if not mask.any():
        reason_counts["empty_raster"] = reason_counts.get("empty_raster", 0) + 1
        return np.zeros((grid_res_y, grid_res_x), dtype=bool)

    return mask


# ------------------------ Phase A / Phase B -------------------------------


@dataclass(frozen=True)
class SupportVolume:
    vol: np.ndarray                 # float (U,Y,X) in [0,1]
    xs: np.ndarray                  # (X,) bin centers
    ys: np.ndarray                  # (Y,) bin centers
    us: np.ndarray                  # (U,) u-centers (log10 time)
    counts_u: np.ndarray            # (U,) number of points per u-slice

    # diagnostics
    valid_slices_mask: np.ndarray   # (U,) bool
    n_invalid_slices: int
    reason_counts: Dict[str, int]


def build_support_volume(
    *,
    df,
    pop: str,
    xcol: str,
    ycol: str,
    tcol: str,
    popcol: str,
    # Tier 1 / Tier 2 (public)
    grid_res_xy: int,
    grid_res_u: int,
    mass_frac: float = 0.90,
    sigma_u: float = 1.0,
    min_points_per_slice: int = 5,
    time_eps: float = 1.0,
    fill_u_gaps: bool = True,
    # global XY edges (single source of truth)
    x_edges: Optional[np.ndarray] = None,
    y_edges: Optional[np.ndarray] = None,
    xy_pad_frac: float = 0.05,
    trim_center: str = "median",
    verbose: bool = False,
    # Advanced (kept minimal; placeholder for future ideas)
    # NOTE: "smooth distance field" is intentionally NOT implemented as a codepath yet.
) -> Optional[SupportVolume]:
    """
    Build support volume for one population using the canonical trimmed convex hull masks.
    """
    sub = df[df[popcol].astype(str) == str(pop)]
    if sub.empty:
        return None

    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    t = sub[tcol].to_numpy(dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    x = x[m]
    y = y[m]
    t = t[m]
    if x.size == 0:
        return None

    grid_res_xy = int(grid_res_xy)
    grid_res_u = int(grid_res_u)

    # ---- XY edges
    if x_edges is None or y_edges is None:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))

        x_pad = float(xy_pad_frac) * (x_max - x_min if x_max > x_min else 1.0)
        y_pad = float(xy_pad_frac) * (y_max - y_min if y_max > y_min else 1.0)

        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        x_edges = np.linspace(x_min, x_max, grid_res_xy + 1)
        y_edges = np.linspace(y_min, y_max, grid_res_xy + 1)
    else:
        x_edges = np.asarray(x_edges, dtype=float)
        y_edges = np.asarray(y_edges, dtype=float)
        if x_edges.ndim != 1 or y_edges.ndim != 1:
            raise ValueError("x_edges and y_edges must be 1D arrays")
        if x_edges.size != grid_res_xy + 1 or y_edges.size != grid_res_xy + 1:
            raise ValueError(
                f"x_edges/y_edges must have length grid_res_xy+1 ({grid_res_xy+1}); "
                f"got {x_edges.size}, {y_edges.size}"
            )
        if not (np.all(np.diff(x_edges) > 0) and np.all(np.diff(y_edges) > 0)):
            raise ValueError("x_edges/y_edges must be strictly increasing")

    xs = 0.5 * (x_edges[:-1] + x_edges[1:])
    ys = 0.5 * (y_edges[:-1] + y_edges[1:])

    # ---- time -> u
    t_pos = np.maximum(t, float(time_eps))
    u = np.log10(t_pos)

    u_min, u_max = float(np.min(u)), float(np.max(u))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        return None

    u_edges = np.linspace(u_min, u_max, grid_res_u + 1)
    us = 0.5 * (u_edges[:-1] + u_edges[1:])

    # assign points to u bins
    u_bin = np.searchsorted(u_edges, u, side="right") - 1
    u_bin = np.clip(u_bin, 0, grid_res_u - 1)

    counts_u = np.bincount(u_bin, minlength=grid_res_u)
    masks = np.zeros((grid_res_u, grid_res_xy, grid_res_xy), dtype=bool)

    reason_counts: Dict[str, int] = {}
    valid_slices = np.zeros((grid_res_u,), dtype=bool)

    # Phase A: per-slice masks
    for ui in range(grid_res_u):
        if counts_u[ui] < int(min_points_per_slice):
            # count in reason_counts inside _mask_xy_for_slice for consistency,
            # but we can do it here too; keep it centralized:
            reason_counts["too_few_points"] = reason_counts.get("too_few_points", 0) + 1
            continue

        sel = (u_bin == ui)
        xsamp = x[sel]
        ysamp = y[sel]

        mask2d = _mask_xy_for_slice(
            xsamp,
            ysamp,
            x_edges=x_edges,
            y_edges=y_edges,
            mass_frac=float(mass_frac),
            trim_center=str(trim_center),
            min_points_per_slice=int(min_points_per_slice),
            reason_counts=reason_counts,
        )

        if mask2d.any():
            masks[ui] = mask2d
            valid_slices[ui] = True
        else:
            # _mask_xy_for_slice already counted the likely cause; keep a generic fallback too
            reason_counts["invalid_slice"] = reason_counts.get("invalid_slice", 0) + 1

    # Optional: fill gaps along u on boolean masks
    masks_pre_fill = masks
    if bool(fill_u_gaps):
        masks = _fill_u_gaps_copy_nearest(masks)

    # Update valid slices based on final boolean slices (after fill)
    valid_slices_after = masks.reshape(grid_res_u, -1).any(axis=1)
    n_invalid = int((~valid_slices_after).sum())

    # Phase B: u-only smoothing => float field in [0,1]
    vol = masks.astype(float)
    if float(sigma_u) > 0:
        vol = gaussian_filter(vol, sigma=(float(sigma_u), 0.0, 0.0), mode="nearest")
    vol = np.clip(vol, 0.0, 1.0)

    if verbose:
        # summarize slice counts
        c = counts_u.astype(int)
        cpos = c[c > 0]
        cmin = int(cpos.min()) if cpos.size else 0
        cmed = float(np.median(cpos)) if cpos.size else 0.0
        cmax = int(cpos.max()) if cpos.size else 0

        v0 = int(masks_pre_fill.reshape(grid_res_u, -1).any(axis=1).sum())
        v1 = int(valid_slices_after.sum())

        # pretty print reasons
        reason_str = ", ".join([f"{k}={v}" for k, v in sorted(reason_counts.items())]) if reason_counts else "none"

        print(
            f"[{pop}] support_volume: valid_slices {v0}->{v1}/{grid_res_u} | "
            f"points_per_slice(min/med/max)={cmin}/{cmed:.1f}/{cmax} | "
            f"mass_frac(keep)={mass_frac:.3f} trim_center={trim_center} | "
            f"reasons: {reason_str}"
        )

    return SupportVolume(
        vol=vol,
        xs=xs,
        ys=ys,
        us=us,
        counts_u=counts_u,
        valid_slices_mask=valid_slices_after.astype(bool),
        n_invalid_slices=n_invalid,
        reason_counts=reason_counts,
    )


# ---------------------------- geometry ------------------------------------


def compute_support_tube_mesh(
    *,
    df,
    pop: str,
    xcol: str,
    ycol: str,
    tcol: str,
    popcol: str,
    # Tier 1 / Tier 2 (public)
    grid_res_xy: int,
    grid_res_u: int,
    mass_frac: float = 0.90,
    sigma_u: float = 1.0,
    min_points_per_slice: int = 5,
    time_eps: float = 1.0,
    x_edges: Optional[np.ndarray] = None,
    y_edges: Optional[np.ndarray] = None,
    trim_center: str = "median",
    verbose: bool = False,
    # Advanced (kept minimal and grouped)
    tau: float = 0.5,
    post_smooth_u: float = 0.0,
    post_smooth_xy: float = 0.0,
    close_r_u: int = 0,
    close_r_xy: int = 0,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute Mesh3d geometry for the population support tube.

    Pipeline:
      1) build_support_volume() => vol in [0,1]
      2) optional post-smoothing in (u,y,x)
      3) threshold at tau => boolean mask
      4) optional morphological closing
      5) marching cubes
      6) map voxel coords -> xs/ys/us -> Z=10**u
    """
    sv = build_support_volume(
        df=df,
        pop=pop,
        xcol=xcol,
        ycol=ycol,
        tcol=tcol,
        popcol=popcol,
        grid_res_xy=int(grid_res_xy),
        grid_res_u=int(grid_res_u),
        mass_frac=float(mass_frac),
        sigma_u=float(sigma_u),
        min_points_per_slice=int(min_points_per_slice),
        time_eps=float(time_eps),
        fill_u_gaps=True,
        x_edges=x_edges,
        y_edges=y_edges,
        trim_center=str(trim_center),
        verbose=bool(verbose),
    )
    if sv is None:
        return None

    vol = sv.vol
    if vol.size == 0 or vol.shape[0] < 2:
        return None

    field = vol.astype(float)

    # Optional post-smoothing (single concept; keep minimal knobs)
    if float(post_smooth_u) > 0 or float(post_smooth_xy) > 0:
        field = gaussian_filter(
            field,
            sigma=(float(post_smooth_u), float(post_smooth_xy), float(post_smooth_xy)),
            mode="nearest",
        )

    mask = field >= float(tau)

    # Optional morphological closing
    if int(close_r_u) > 0 or int(close_r_xy) > 0:
        if int(close_r_u) > 0 and int(close_r_xy) > 0:
            r = int(max(close_r_u, close_r_xy))
            mask = binary_closing(mask, structure=_make_ball(r))
        elif int(close_r_xy) > 0:
            st = _make_disk(int(close_r_xy))
            for ui in range(mask.shape[0]):
                mask[ui] = binary_closing(mask[ui], structure=st)
        elif int(close_r_u) > 0:
            r = int(close_r_u)
            structure = np.ones((2 * r + 1, 1, 1), dtype=bool)
            mask = binary_closing(mask, structure=structure)

    if not mask.any():
        if verbose:
            print(f"[{pop}] support_tube_mesh: empty after threshold/closing")
        return None

    scalar = mask.astype(np.float32)

    try:
        verts, faces, _, _ = marching_cubes(scalar, level=0.5)
    except RuntimeError as e:
        if verbose:
            print(f"[{pop}] marching_cubes failed: {e}")
        return None

    u_idx = verts[:, 0]
    y_idx = verts[:, 1]
    x_idx = verts[:, 2]

    X = _interp_centers(x_idx, sv.xs)
    Y = _interp_centers(y_idx, sv.ys)
    U = _interp_centers(u_idx, sv.us)
    Z = 10.0 ** U

    return {
        "X": X.astype(float),
        "Y": Y.astype(float),
        "Z": Z.astype(float),
        "faces": faces.astype(np.int32),
    }


def compute_support_tube_rings(
    *,
    df,
    pop: str,
    xcol: str,
    ycol: str,
    tcol: str,
    popcol: str,
    # Tier 1 / Tier 2 (public)
    grid_res_xy: int,
    grid_res_u: int,
    mass_frac: float = 0.90,
    sigma_u: float = 1.0,
    min_points_per_slice: int = 5,
    time_eps: float = 1.0,
    x_edges: Optional[np.ndarray] = None,
    y_edges: Optional[np.ndarray] = None,
    trim_center: str = "median",
    # ring binning (public; these control only how rings are grouped)
    nbins_old: int = 24,
    recent_bin_width: int = 1,
    override_window: int = 12,
    override_valid_frac: float = 0.7,
    min_boundary_points: int = 50,
    verbose: bool = False,
    # Advanced (grouped)
    tau: float = 0.5,
    tau_recent: float = 0.5,
) -> List[Dict[str, np.ndarray]]:
    """
    Build ring outlines (polylines) from the support mask volume.
    """
    sv = build_support_volume(
        df=df,
        pop=pop,
        xcol=xcol,
        ycol=ycol,
        tcol=tcol,
        popcol=popcol,
        grid_res_xy=int(grid_res_xy),
        grid_res_u=int(grid_res_u),
        mass_frac=float(mass_frac),
        sigma_u=float(sigma_u),
        min_points_per_slice=int(min_points_per_slice),
        time_eps=float(time_eps),
        fill_u_gaps=True,
        x_edges=x_edges,
        y_edges=y_edges,
        trim_center=str(trim_center),
        verbose=bool(verbose),
    )
    if sv is None:
        return []

    vol = sv.vol
    U, Y, X = vol.shape
    if U < 2:
        return []

    mask_old = vol >= float(tau)
    mask_recent = vol >= float(tau_recent)

    def mask2d_to_poly(mask2d: np.ndarray) -> Optional[np.ndarray]:
        if mask2d is None or not mask2d.any():
            return None
        contours = find_contours(mask2d.astype(float), 0.5)
        c = _largest_contour(contours)
        if c is None:
            return None
        if c.shape[0] < int(min_boundary_points):
            return None
        y_idx = c[:, 0]
        x_idx = c[:, 1]
        xs = _interp_centers(x_idx, sv.xs)
        ys = _interp_centers(y_idx, sv.ys)
        poly = np.stack([xs, ys], axis=1)
        if poly.shape[0] > 0:
            poly = np.vstack([poly, poly[0:1]])
        return poly

    w = int(max(1, override_window))
    w = min(w, U)
    recent_start = U - w

    valid_recent = 0
    for ui in range(recent_start, U):
        poly = mask2d_to_poly(mask_recent[ui])
        if poly is not None:
            valid_recent += 1

    recent_frac = valid_recent / max(w, 1)
    use_recent = recent_frac >= float(override_valid_frac)

    if verbose:
        print(
            f"[{pop}] rings: recent_window={w} valid_recent={valid_recent}/{w} "
            f"(frac={recent_frac:.2f}) use_recent={use_recent}"
        )

    rings: List[Dict[str, np.ndarray]] = []

    old_end = recent_start if use_recent else U
    old_end = max(old_end, 0)

    if old_end >= 2 and int(nbins_old) > 0:
        nb = int(max(1, nbins_old))
        edges = np.linspace(0, old_end, nb + 1).astype(int)
        edges[0] = 0
        edges[-1] = old_end
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([0, old_end], dtype=int)

        for bi in range(len(edges) - 1):
            a, b = int(edges[bi]), int(edges[bi + 1])
            if b <= a:
                continue
            m = np.any(mask_old[a:b], axis=0)
            poly = mask2d_to_poly(m)
            if poly is None:
                continue
            u_center = float(np.mean(sv.us[a:b]))
            z = float(10.0 ** u_center)
            rings.append(
                dict(
                    x=poly[:, 0],
                    y=poly[:, 1],
                    z=np.full(poly.shape[0], z, dtype=float),
                    kind="old",
                    hovertemplate=f"{pop} old bin<br>Time: {z:.3g}<extra></extra>",
                )
            )

    if use_recent and w >= 1:
        bw = int(max(1, recent_bin_width))
        for a in range(recent_start, U, bw):
            b = min(U, a + bw)
            m = np.any(mask_recent[a:b], axis=0)
            poly = mask2d_to_poly(m)
            if poly is None:
                continue
            u_center = float(np.mean(sv.us[a:b]))
            z = float(10.0 ** u_center)
            rings.append(
                dict(
                    x=poly[:, 0],
                    y=poly[:, 1],
                    z=np.full(poly.shape[0], z, dtype=float),
                    kind="recent",
                    hovertemplate=f"{pop} recent bin<br>Time: {z:.3g}<extra></extra>",
                )
            )

    return rings