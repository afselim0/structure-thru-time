#!/usr/bin/env python3
"""
src/support_tube.py

Support-tube geometry builder for PlotEmbeddings.

Implements:
- Phase A: per-u slice (u=log10(time)) mass-fraction support mask
- Phase B: time/u smoothing + optional morphological closing
- Geometry:
    - compute_support_tube_mesh() -> Mesh3d vertices + faces
    - compute_support_tube_rings() -> Scatter3d ring polylines for time-binned outlines

Conventions
-----------
Grid arrays:
- XY grid is (Y, X) with y as rows, x as cols (matches np.histogram2d usage)
- Volume is (U, Y, X) where U indexes log10(time) bins.

Mapping:
- u = log10(max(time, time_eps))
- For plotting: z = 10**u (so z is in original time units, suitable for log z-axis)

Dependencies:
- numpy
- scipy.ndimage (gaussian_filter, binary_closing)
- skimage.measure (marching_cubes, find_contours)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing
from skimage.measure import marching_cubes, find_contours


# ---------------------------- utilities -----------------------------------


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _mass_fraction_threshold(density_2d: np.ndarray, mass_frac: float) -> float:
    """
    Return threshold tau such that pixels with value >= tau contain `mass_frac`
    of total mass (sum of density).
    """
    mass_frac = float(mass_frac)
    if not (0.0 < mass_frac <= 1.0):
        raise ValueError(f"mass_frac must be in (0,1], got {mass_frac}")

    d = np.asarray(density_2d, dtype=float).ravel()
    total = float(d.sum())
    if total <= 0:
        return np.inf  # yields empty mask

    order = np.argsort(d)[::-1]
    vals = d[order]
    csum = np.cumsum(vals)

    cutoff = mass_frac * total
    idx = int(np.searchsorted(csum, cutoff, side="left"))
    idx = int(np.clip(idx, 0, len(vals) - 1))
    return float(vals[idx])


def _interp_centers(idx: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    idx: float array in [0, n-1]
    centers: (n,) array
    """
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
    """
    Binary structuring element disk in 2D.
    """
    r = int(radius)
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= (r * r)


def _make_ball(radius: int) -> np.ndarray:
    """
    Binary structuring element ball in 3D (U,Y,X).
    """
    r = int(radius)
    if r <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy + zz * zz) <= (r * r)


# ------------------------ Phase A / Phase B -------------------------------


@dataclass(frozen=True)
class SupportVolume:
    vol: np.ndarray      # float (U,Y,X) in [0,1]
    xs: np.ndarray       # (X,) bin centers
    ys: np.ndarray       # (Y,) bin centers
    us: np.ndarray       # (U,) u-centers (log10 time)
    counts_u: np.ndarray # (U,) number of points per u-slice


def build_support_volume(
    *,
    df,
    pop: str,
    xcol: str,
    ycol: str,
    tcol: str,
    popcol: str,
    grid_res_xy: int,
    grid_res_u: int,
    mass_frac: float,
    sigma_xy: float,
    sigma_u: float,
    min_points_per_slice: int,
    time_eps: float,
    xy_pad_frac: float = 0.05,
    verbose: bool = False,
) -> Optional[SupportVolume]:
    """
    Phase A:
      For each u slice, compute 2D histogram H(y,x), smooth with sigma_xy,
      then threshold to mass-fraction mask.

    Phase B:
      Smooth masks along u with sigma_u (gaussian on axis 0).
      Output vol float in [0,1].
    """
    sub = df[df[popcol].astype(str) == str(pop)]
    if sub.empty:
        return None

    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    t = sub[tcol].to_numpy(dtype=float)

    x = _finite(x)
    y = _finite(y)
    t = _finite(t)

    if x.size == 0 or y.size == 0 or t.size == 0:
        return None

    # ---- XY bounds + padding
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    x_pad = xy_pad_frac * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = xy_pad_frac * (y_max - y_min if y_max > y_min else 1.0)

    x_min -= x_pad; x_max += x_pad
    y_min -= y_pad; y_max += y_pad

    x_edges = np.linspace(x_min, x_max, int(grid_res_xy) + 1)
    y_edges = np.linspace(y_min, y_max, int(grid_res_xy) + 1)

    xs = 0.5 * (x_edges[:-1] + x_edges[1:])
    ys = 0.5 * (y_edges[:-1] + y_edges[1:])

    # ---- time -> u
    t_pos = np.maximum(t, float(time_eps))
    u = np.log10(t_pos)

    u_min, u_max = float(np.min(u)), float(np.max(u))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        return None

    u_edges = np.linspace(u_min, u_max, int(grid_res_u) + 1)
    us = 0.5 * (u_edges[:-1] + u_edges[1:])

    # assign points to u bins
    u_bin = np.searchsorted(u_edges, u, side="right") - 1
    u_bin = np.clip(u_bin, 0, int(grid_res_u) - 1)

    counts_u = np.bincount(u_bin, minlength=int(grid_res_u))
    masks = np.zeros((int(grid_res_u), int(grid_res_xy), int(grid_res_xy)), dtype=bool)

    if verbose:
        empty = int((counts_u < int(min_points_per_slice)).sum())
        print(
            f"[{pop}] build_support_volume: grid=(U={grid_res_u},Y=X={grid_res_xy}) "
            f"u_range=[{u_min:.3f},{u_max:.3f}] "
            f"sigma_xy={sigma_xy} sigma_u={sigma_u} mass_frac={mass_frac} "
            f"slices(<{min_points_per_slice})={empty}/{grid_res_u}"
        )

    # Phase A: per-slice mask
    for ui in range(int(grid_res_u)):
        if counts_u[ui] < int(min_points_per_slice):
            continue

        sel = (u_bin == ui)
        xsamp = x[sel]
        ysamp = y[sel]

        H, _, _ = np.histogram2d(
            ysamp, xsamp,  # rows=y, cols=x
            bins=(y_edges, x_edges),
        )
        H = H.astype(float)

        if float(sigma_xy) > 0:
            Hs = gaussian_filter(H, sigma=(float(sigma_xy), float(sigma_xy)), mode="nearest")
        else:
            Hs = H

        tau = _mass_fraction_threshold(Hs, float(mass_frac))
        if not np.isfinite(tau):
            continue

        masks[ui] = (Hs >= tau)

    # Phase B: smooth along u only
    vol = masks.astype(float)
    if float(sigma_u) > 0:
        vol = gaussian_filter(vol, sigma=(float(sigma_u), 0.0, 0.0), mode="nearest")

    vol = np.clip(vol, 0.0, 1.0)

    return SupportVolume(vol=vol, xs=xs, ys=ys, us=us, counts_u=counts_u)


# ---------------------------- geometry ------------------------------------


def compute_support_tube_mesh(
    *,
    df,
    pop: str,
    xcol: str,
    ycol: str,
    tcol: str,
    popcol: str,
    grid_res_xy: int,
    grid_res_u: int,
    mass_frac: float,
    sigma_xy: float,
    sigma_u: float,
    min_points_per_slice: int,
    time_eps: float,
    tau: float = 0.5,
    u_smooth_sigma: float = 0.0,
    xy_smooth_sigma: float = 0.0,
    close_r_u: int = 0,
    close_r_xy: int = 0,
    verbose: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Returns:
      dict with keys:
        - X, Y, Z: (N,) vertex coordinates in real space (Z is time)
        - faces:   (F,3) int array of triangle indices

    Strategy:
      1) build_support_volume (vol in [0,1])
      2) optionally smooth scalar field in (u,y,x)
      3) threshold at tau into boolean mask
      4) optional morphological closing (3D or per-slice)
      5) marching_cubes on mask.astype(float) at level=0.5
      6) map voxel coords -> xs/ys/us -> Z=10**u
    """
    sv = build_support_volume(
        df=df,
        pop=pop,
        xcol=xcol,
        ycol=ycol,
        tcol=tcol,
        popcol=popcol,
        grid_res_xy=grid_res_xy,
        grid_res_u=grid_res_u,
        mass_frac=mass_frac,
        sigma_xy=sigma_xy,
        sigma_u=sigma_u,
        min_points_per_slice=min_points_per_slice,
        time_eps=time_eps,
        verbose=verbose,
    )
    if sv is None:
        return None

    vol = sv.vol
    if vol.size == 0 or vol.shape[0] < 2:
        return None

    field = vol.astype(float)

    # Optional additional smoothing (separate from sigma_u/sigma_xy used in mask construction)
    if float(u_smooth_sigma) > 0 or float(xy_smooth_sigma) > 0:
        field = gaussian_filter(
            field,
            sigma=(float(u_smooth_sigma), float(xy_smooth_sigma), float(xy_smooth_sigma)),
            mode="nearest",
        )

    # Threshold to boolean
    mask = field >= float(tau)

    # Optional morphological closing
    if int(close_r_u) > 0 or int(close_r_xy) > 0:
        if int(close_r_u) > 0 and int(close_r_xy) > 0:
            # full 3D closing with ball
            r = int(max(close_r_u, close_r_xy))
            mask = binary_closing(mask, structure=_make_ball(r))
        else:
            # per-slice 2D closing in XY (common useful case)
            if int(close_r_xy) > 0:
                st = _make_disk(int(close_r_xy))
                for ui in range(mask.shape[0]):
                    mask[ui] = binary_closing(mask[ui], structure=st)

    if not mask.any():
        if verbose:
            print(f"[{pop}] compute_support_tube_mesh: empty mask after threshold/closing")
        return None

    # Marching cubes expects scalar volume
    scalar = mask.astype(np.float32)

    try:
        verts, faces, _, _ = marching_cubes(scalar, level=0.5)
    except RuntimeError as e:
        if verbose:
            print(f"[{pop}] marching_cubes failed: {e}")
        return None

    # verts columns correspond to (u_index, y_index, x_index) in scalar[U,Y,X]
    u_idx = verts[:, 0]
    y_idx = verts[:, 1]
    x_idx = verts[:, 2]

    # map to real coords using centers (linear interp)
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
    grid_res_xy: int,
    grid_res_u: int,
    mass_frac: float,
    sigma_xy: float,
    sigma_u: float,
    min_points_per_slice: int,
    time_eps: float,
    nbins_old: int = 24,
    recent_bin_width: int = 1,
    override_window: int = 12,
    override_valid_frac: float = 0.7,
    tau: float = 0.5,
    tau_recent: float = 0.5,
    min_boundary_points: int = 50,
    verbose: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """
    Build ring outlines (polylines) from the support mask.

    We first construct the support volume vol in [0,1], then:
    - Old period: combine earlier u-slices into nbins_old bins (OR masks within bin)
    - Recent period: last override_window u-slices are binned by recent_bin_width (finer),
      but only applied if enough of those slices are "valid" (non-empty boundary) according
      to override_valid_frac.

    Returns:
      list of dicts with keys:
        - x, y, z: polyline arrays (z constant)
        - kind: "old" | "recent"
        - hovertemplate: optional string
    """
    sv = build_support_volume(
        df=df,
        pop=pop,
        xcol=xcol,
        ycol=ycol,
        tcol=tcol,
        popcol=popcol,
        grid_res_xy=grid_res_xy,
        grid_res_u=grid_res_u,
        mass_frac=mass_frac,
        sigma_xy=sigma_xy,
        sigma_u=sigma_u,
        min_points_per_slice=min_points_per_slice,
        time_eps=time_eps,
        verbose=verbose,
    )
    if sv is None:
        return []

    vol = sv.vol
    U, Y, X = vol.shape
    if U < 2:
        return []

    # Threshold into boolean masks for old and recent (recent can be more inclusive/strict)
    mask_old = vol >= float(tau)
    mask_recent = vol >= float(tau_recent)

    # Helper: turn 2D mask into a contour polyline in real coords
    def mask2d_to_poly(mask2d: np.ndarray) -> Optional[np.ndarray]:
        if mask2d is None or not mask2d.any():
            return None

        # contours expects 2D float image; extract boundary at 0.5
        contours = find_contours(mask2d.astype(float), 0.5)
        c = _largest_contour(contours)
        if c is None:
            return None
        if c.shape[0] < int(min_boundary_points):
            return None

        # c is (n_points, 2) in (row=y_idx, col=x_idx)
        y_idx = c[:, 0]
        x_idx = c[:, 1]
        xs = _interp_centers(x_idx, sv.xs)
        ys = _interp_centers(y_idx, sv.ys)

        poly = np.stack([xs, ys], axis=1)

        # close the loop
        if poly.shape[0] > 0:
            poly = np.vstack([poly, poly[0:1]])

        return poly

    # Decide if recent override is used
    w = int(max(1, override_window))
    w = min(w, U)
    recent_start = U - w

    # validity: does each slice have a sufficiently long contour?
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

    # --------- OLD bins (everything before recent_start if recent is enabled) ----------
    old_end = recent_start if use_recent else U
    old_end = max(old_end, 0)

    if old_end >= 2 and int(nbins_old) > 0:
        nb = int(max(1, nbins_old))
        edges = np.linspace(0, old_end, nb + 1).astype(int)
        # ensure unique edges
        edges[0] = 0
        edges[-1] = old_end
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([0, old_end], dtype=int)

        for bi in range(len(edges) - 1):
            a, b = int(edges[bi]), int(edges[bi + 1])
            if b <= a:
                continue

            # OR masks across the bin
            m = np.any(mask_old[a:b], axis=0)
            poly = mask2d_to_poly(m)
            if poly is None:
                continue

            # z at bin center (in u)
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

    # --------- RECENT bins (finer in last window) ----------
    if use_recent and w >= 1:
        bw = int(max(1, recent_bin_width))
        start = recent_start

        for a in range(start, U, bw):
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